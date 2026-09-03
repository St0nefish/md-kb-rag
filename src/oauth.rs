//! OAuth 2.1 resource-server support: JWT access-token validation and RFC 9728
//! protected-resource metadata.
//!
//! This module makes the process a *resource server* only. It never issues,
//! refreshes, revokes or introspects tokens, and it never talks to the
//! authorization server except to fetch its public signing keys (JWKS). Everything
//! it needs is either config (`mcp.oauth`, see `config::OAuthConfig`) or derived
//! from it once at construction.
//!
//! ## Dual-mode auth
//!
//! This does NOT replace the static bearer token. `server::bearer_auth` accepts
//! *either* credential: the constant-time static-token comparison it has always
//! done, or a JWT validated here. Claude Code authenticates with the static token
//! and is not being migrated; the OAuth path exists so Claude's hosted surfaces
//! (claude.ai / Desktop / iOS) can connect as a custom connector, which requires an
//! authorization-code flow. A deployment can run either, both, or neither.
//!
//! ## What is deliberately NOT here
//!
//! Per-tool write-scope enforcement — requiring `mcp:write` for `write_document` /
//! `delete_document` / `update_schema` — is **not implemented**. The auth
//! middleware sits in front of the whole `/mcp` endpoint and cannot see which MCP
//! tool a request invokes; that lives in the JSON-RPC body, which only rmcp's
//! transport parses. **Any token that passes validation here currently grants full
//! access, writes included.** The validated scope list is put into request
//! extensions as an [`AuthorizedToken`] precisely so a later change can enforce
//! per-tool scopes at the tool-dispatch layer without re-deriving them.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use jsonwebtoken::jwk::{AlgorithmParameters, JwkSet};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde::Deserialize;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::config::ResolvedOAuthConfig;

/// Path suffix RFC 9728 §3 splices between a resource's authority and its path to
/// form the metadata URL. Also the literal route prefix `server::assemble_router`
/// registers — see `OAuthValidator::new`, which warns when a configured `resource`
/// would derive a URL those routes do not answer on.
pub const PROTECTED_RESOURCE_METADATA_PREFIX: &str = "/.well-known/oauth-protected-resource";

/// How long an unknown `kid` is allowed to trigger a JWKS refetch again.
///
/// An unknown `kid` is attacker-controllable — it is just a field in an unverified
/// token header — so without this a stream of junk tokens would turn this server
/// into an amplifier pointed at the identity provider. One refetch per minute is
/// far faster than any real key rotation needs (JWKS rollovers publish the new key
/// alongside the old one well before signing with it) and slow enough that the IdP
/// never notices us.
const JWKS_MIN_REFETCH_INTERVAL: Duration = Duration::from_secs(60);

/// Ceiling on a single JWKS fetch. Bounds how long the write lock below is held,
/// and therefore how long a stalled IdP can stall token validation.
const JWKS_FETCH_TIMEOUT: Duration = Duration::from_secs(10);

/// Clock-skew allowance on `exp`. Small on purpose: it is there for a few seconds
/// of drift between the IdP's clock and ours, not to extend a token's life.
const CLOCK_SKEW_LEEWAY_SECS: u64 = 60;

/// A successfully validated access token, inserted into request extensions by
/// `server::bearer_auth`.
///
/// Nothing reads `scopes` yet — see the module doc: write-scope enforcement is not
/// implemented, and every valid token currently grants full access. This type
/// exists so that when it is, the scopes come from the one place that actually
/// verified them rather than being re-parsed out of the header a second time.
#[derive(Debug, Clone)]
pub struct AuthorizedToken {
    /// The token's `sub`, when it carried one. Useful for request logging.
    pub subject: Option<String>,
    /// The `scope` claim, split on whitespace.
    pub scopes: Vec<String>,
}

impl AuthorizedToken {
    /// Whether the token carries `scope`. Unused today (nothing enforces per-tool
    /// scopes yet — see the module doc); kept as the single place that will answer
    /// that question so callers never hand-roll a `.iter().any()` over `scopes`.
    #[allow(dead_code)]
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope)
    }
}

/// Why a bearer credential was refused, and — crucially — with which HTTP status.
///
/// The split is the whole point: RFC 6750 distinguishes "this token is not good"
/// (401, go get a new one) from "this token is fine but not sufficient" (403, ask
/// for more scope). A client that gets 401 for an insufficient-scope token will
/// loop through the authorization flow forever and land back on the same refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenRejection {
    /// 401 `invalid_token`: malformed, unsigned, wrong issuer/audience, expired, or
    /// signed by a key we could not obtain. The string is for logs only — it is
    /// never returned to the caller, since telling an unauthenticated client
    /// exactly which check failed is a free oracle.
    Invalid(String),
    /// 403 `insufficient_scope`: signature, issuer, audience and expiry all passed,
    /// but the token does not carry the configured required scope.
    InsufficientScope,
}

/// The subset of an access token's claims this server reads.
///
/// `iss`, `aud` and `exp` are deliberately absent: `jsonwebtoken::decode` validates
/// all three against the [`Validation`] built in [`OAuthValidator::new`] before this
/// struct is ever populated, and duplicating them here would invite a second,
/// weaker check drifting alongside the real one.
#[derive(Debug, Deserialize)]
struct AccessTokenClaims {
    #[serde(default)]
    sub: Option<String>,
    /// OAuth 2.0's space-separated scope string (RFC 6749 §3.3), which is the shape
    /// Authentik issues. Absent on a token minted without any scope, which is not
    /// an error here — it just fails the `required_scope` check below with a 403.
    #[serde(default)]
    scope: Option<String>,
}

/// The in-memory JWKS, plus when we last *attempted* to refresh it.
///
/// Attempt, not success, on purpose: a failing IdP must be backed off exactly like
/// a successful-but-stale one, or an outage turns every junk token into a retry
/// against a service that is already struggling.
#[derive(Default)]
struct JwksCache {
    keys: HashMap<String, DecodingKey>,
    last_attempt: Option<Instant>,
}

pub struct OAuthValidator {
    /// Everything below is derived from this once; it is kept for the metadata
    /// document and for logging.
    config: ResolvedOAuthConfig,
    /// Pre-rendered so the 401/403 paths are a string clone, not a `format!` per
    /// rejected request.
    resource_metadata_url: String,
    /// `scopes_supported`, space-joined, for the 401 challenge's `scope` parameter.
    supported_scopes: String,
    /// The RFC 9728 document both well-known routes serve, rendered once.
    metadata: serde_json::Value,
    /// Issuer/audience/expiry/algorithm policy, built once. `jsonwebtoken` applies
    /// all of it inside `decode`, which is what keeps signature verification and
    /// claim validation from being two separately-forgettable steps.
    validation: Validation,
    http: reqwest::Client,
    jwks: RwLock<JwksCache>,
    /// Normally [`JWKS_MIN_REFETCH_INTERVAL`]; overridden only by tests, which
    /// would otherwise have to sleep a minute to observe a refetch.
    jwks_min_refetch_interval: Duration,
}

impl OAuthValidator {
    pub fn new(config: &ResolvedOAuthConfig) -> Result<Self> {
        Self::build(config, JWKS_MIN_REFETCH_INTERVAL)
    }

    fn build(config: &ResolvedOAuthConfig, jwks_min_refetch_interval: Duration) -> Result<Self> {
        let mut validation = Validation::new(Algorithm::RS256);
        // Byte-exact issuer match. Authentik's issuer ends in a slash and the
        // difference matters — `.../mcp-kb-rag/` and `.../mcp-kb-rag` are different
        // strings and only one of them is in the tokens.
        validation.set_issuer(&[&config.issuer]);
        // The audience is the OAuth CLIENT ID, not this server's resource URL.
        // Authentik does not implement RFC 8707 resource indicators: it ignores the
        // `resource` request parameter and hardcodes `aud` to the provider's
        // client_id. That is a known, accepted deviation for this deployment. Do not
        // "correct" this to `config.resource` — every real token would start
        // failing. `jsonwebtoken` accepts `aud` as either a string or an array of
        // strings and treats an array as a match if any element matches, which is
        // what RFC 7519 §4.1.3 asks for.
        validation.set_audience(&[&config.audience]);
        // `jsonwebtoken` only validates `iss`/`aud` when the claim is *present*, so
        // requiring them here is what turns "wrong issuer" and "no issuer at all"
        // into the same refusal. Without this a token carrying neither claim would
        // sail through both checks.
        validation.set_required_spec_claims(&["exp", "iss", "aud"]);
        validation.leeway = CLOCK_SKEW_LEEWAY_SECS;
        validation.validate_exp = true;
        // RS256 only — the single algorithm the AS actually signs with. Pinning it
        // is what makes the classic `alg: none` / HS256-with-the-public-key
        // confusions unrepresentable rather than merely unlikely.
        validation.algorithms = vec![Algorithm::RS256];

        let resource_metadata_url = resource_metadata_url(&config.resource);
        let supported_scopes = config.scopes_supported.join(" ");

        // Warn rather than fail: the two well-known routes the server registers are
        // fixed (`/.well-known/oauth-protected-resource` and `.../mcp`), because
        // that is where this deployment's MCP endpoint lives and where clients
        // probe. A `resource` with some other path still produces a valid metadata
        // document, but the URL advertised in `WWW-Authenticate` would point at a
        // path this process does not answer on — worth saying out loud once at
        // startup, not worth refusing to boot over.
        let advertised_path = resource_metadata_url
            .split_once("://")
            .and_then(|(_, rest)| rest.find('/').map(|i| rest[i..].to_string()))
            .unwrap_or_default();
        if advertised_path != format!("{PROTECTED_RESOURCE_METADATA_PREFIX}/mcp")
            && advertised_path != PROTECTED_RESOURCE_METADATA_PREFIX
        {
            warn!(
                resource = %config.resource,
                advertised = %resource_metadata_url,
                "mcp.oauth.resource derives a protected-resource metadata URL this server \
                 does not serve — it only answers on {PROTECTED_RESOURCE_METADATA_PREFIX} \
                 and {PROTECTED_RESOURCE_METADATA_PREFIX}/mcp. OAuth clients will 404 on \
                 discovery."
            );
        }

        let metadata = serde_json::json!({
            "resource": config.resource,
            // Echoed byte-identically. A client compares this against the `iss` of
            // the tokens it receives, so normalizing (adding or trimming a trailing
            // slash, lowercasing) here would break that comparison.
            "authorization_servers": [config.issuer],
            "scopes_supported": config.scopes_supported,
            "bearer_methods_supported": ["header"],
            "resource_name": "mcp-md-wiki knowledge base (MCP)",
        });

        let http = reqwest::Client::builder()
            .timeout(JWKS_FETCH_TIMEOUT)
            .build()
            .context("Failed to build the HTTP client for JWKS fetches")?;

        Ok(Self {
            config: config.clone(),
            resource_metadata_url,
            supported_scopes,
            metadata,
            validation,
            http,
            jwks: RwLock::new(JwksCache::default()),
            jwks_min_refetch_interval,
        })
    }

    /// The RFC 9728 document, served by both well-known routes.
    pub fn metadata(&self) -> serde_json::Value {
        self.metadata.clone()
    }

    /// The `WWW-Authenticate` value for a 401.
    ///
    /// Load-bearing, not cosmetic: claude.ai has been observed refusing to start the
    /// authorization flow at all when a 401 arrives without it, because
    /// `resource_metadata` is how the client finds the authorization server in the
    /// first place. Claude Code tolerates its absence, which is exactly why it is
    /// easy to drop and hard to notice. Emitted on EVERY 401 once OAuth is
    /// configured — including a failed static-bearer request, since the server
    /// cannot tell which credential the caller meant to present.
    pub fn invalid_token_challenge(&self) -> String {
        format!(
            "Bearer error=\"invalid_token\", resource_metadata=\"{}\", scope=\"{}\"",
            quoted(&self.resource_metadata_url),
            quoted(&self.supported_scopes)
        )
    }

    /// The `WWW-Authenticate` value for a 403: the token was genuinely valid, so
    /// `scope` names what it is *missing* rather than everything on offer. That is
    /// the difference that lets a client re-authorize for the right thing instead of
    /// replaying the same request.
    pub fn insufficient_scope_challenge(&self) -> String {
        format!(
            "Bearer error=\"insufficient_scope\", scope=\"{}\", resource_metadata=\"{}\"",
            quoted(&self.config.required_scope),
            quoted(&self.resource_metadata_url)
        )
    }

    /// Validate a bearer credential as a JWT access token.
    ///
    /// Order matters and is the RFC's: signature first (nothing in an unverified
    /// token may be trusted, including the claims the later checks read), then
    /// issuer / audience / expiry — all three inside `jsonwebtoken::decode`, so they
    /// cannot be reordered ahead of the signature by accident — then scope.
    pub async fn validate(&self, token: &str) -> Result<AuthorizedToken, TokenRejection> {
        // The header is unverified data. It is read only to pick which key to
        // verify WITH; nothing from it is trusted afterwards, and `alg` is checked
        // against our pinned list rather than obeyed.
        let header = decode_header(token)
            .map_err(|e| TokenRejection::Invalid(format!("malformed token header: {e}")))?;
        if header.alg != Algorithm::RS256 {
            return Err(TokenRejection::Invalid(format!(
                "unsupported token algorithm {:?}, only RS256 is accepted",
                header.alg
            )));
        }

        let key = self.decoding_key(header.kid.as_deref()).await?;

        let data = decode::<AccessTokenClaims>(token, &key, &self.validation).map_err(|e| {
            // `jsonwebtoken`'s error kinds already distinguish bad signature from
            // bad issuer/audience/expiry; all of them are 401 `invalid_token` to the
            // caller, and only the log gets to know which.
            TokenRejection::Invalid(format!("token rejected: {e}"))
        })?;

        let scopes: Vec<String> = data
            .claims
            .scope
            .as_deref()
            .unwrap_or("")
            .split_whitespace()
            .map(str::to_string)
            .collect();

        if !scopes.iter().any(|s| s == &self.config.required_scope) {
            debug!(
                required = %self.config.required_scope,
                present = ?scopes,
                "OAuth token valid but missing the required scope"
            );
            return Err(TokenRejection::InsufficientScope);
        }

        Ok(AuthorizedToken {
            subject: data.claims.sub,
            scopes,
        })
    }

    /// Resolve the signing key for `kid`, fetching or refetching the JWKS as needed.
    ///
    /// Fails closed in every failure mode — an unreachable IdP, a malformed key set,
    /// an unknown `kid` during the refetch cooldown — because the alternative shape
    /// ("could not check, so allow") is the one bug in this file that would be worth
    /// a CVE.
    async fn decoding_key(&self, kid: Option<&str>) -> Result<DecodingKey, TokenRejection> {
        if let Some(key) = self.cached_key(kid).await {
            return Ok(key);
        }

        // The write lock is held across the fetch on purpose. It serializes a
        // thundering herd of concurrent first-use requests into one HTTP call
        // instead of N, and `JWKS_FETCH_TIMEOUT` bounds how long everyone else
        // waits. The alternative — dropping the lock to fetch — buys nothing here
        // because there is nothing useful for the other waiters to do meanwhile:
        // without keys they can only fail.
        let mut cache = self.jwks.write().await;

        // Another task may have fetched while we waited for the lock.
        if let Some(key) = lookup(&cache.keys, kid) {
            return Ok(key);
        }

        if let Some(last) = cache.last_attempt
            && last.elapsed() < self.jwks_min_refetch_interval
        {
            // See `JWKS_MIN_REFETCH_INTERVAL`: `kid` comes from an unverified token
            // header, so an unknown one must not be able to schedule IdP traffic.
            return Err(TokenRejection::Invalid(format!(
                "unknown key id {kid:?} and the JWKS was refetched less than {}s ago",
                self.jwks_min_refetch_interval.as_secs()
            )));
        }

        cache.last_attempt = Some(Instant::now());
        match self.fetch_jwks().await {
            Ok(keys) => {
                debug!(
                    count = keys.len(),
                    jwks_uri = %self.config.jwks_uri,
                    "Fetched JWKS"
                );
                cache.keys = keys;
            }
            Err(e) => {
                // Deliberately keep the previous `keys` map: a transient IdP outage
                // should not invalidate keys that are still perfectly good.
                warn!(
                    jwks_uri = %self.config.jwks_uri,
                    error = %e,
                    "JWKS fetch failed — tokens signed by a key we do not already \
                     hold will be rejected until the next attempt"
                );
                return Err(TokenRejection::Invalid(format!("JWKS fetch failed: {e}")));
            }
        }

        lookup(&cache.keys, kid).ok_or_else(|| {
            TokenRejection::Invalid(format!("no key matching kid {kid:?} in the fetched JWKS"))
        })
    }

    async fn cached_key(&self, kid: Option<&str>) -> Option<DecodingKey> {
        let cache = self.jwks.read().await;
        lookup(&cache.keys, kid)
    }

    async fn fetch_jwks(&self) -> Result<HashMap<String, DecodingKey>> {
        let resp = self
            .http
            .get(&self.config.jwks_uri)
            .send()
            .await
            .context("request failed")?
            .error_for_status()
            .context("non-success status")?;
        let set: JwkSet = resp.json().await.context("response was not a JWK Set")?;

        let mut keys = HashMap::new();
        for jwk in &set.keys {
            // Only RSA keys are usable here — `validation.algorithms` is RS256-only,
            // so an EC or OKP key in the set is not something we could verify with
            // even if a token named it. Skipped rather than fatal: a mixed key set
            // is a perfectly normal thing for an AS to publish.
            if !matches!(jwk.algorithm, AlgorithmParameters::RSA(_)) {
                continue;
            }
            let Some(kid) = jwk.common.key_id.clone() else {
                continue;
            };
            match DecodingKey::from_jwk(jwk) {
                Ok(key) => {
                    keys.insert(kid, key);
                }
                Err(e) => warn!(kid = %kid, error = %e, "Skipping unusable JWKS entry"),
            }
        }
        if keys.is_empty() {
            anyhow::bail!("JWK Set contained no usable RSA keys with a kid");
        }
        Ok(keys)
    }
}

/// Find the key for `kid`.
///
/// A token header with no `kid` falls back to the key set's single entry, when
/// there is exactly one. That is not laxity: with one published key there is
/// exactly one key the signature could have been made with, so the fallback picks
/// the same key an explicit `kid` would have. With two or more it refuses rather
/// than trying each, which would turn key rotation into a signature-verification
/// oracle.
fn lookup(keys: &HashMap<String, DecodingKey>, kid: Option<&str>) -> Option<DecodingKey> {
    match kid {
        Some(kid) => keys.get(kid).cloned(),
        None if keys.len() == 1 => keys.values().next().cloned(),
        None => None,
    }
}

/// Escape a value for an HTTP `quoted-string` (RFC 9110 §5.6.4).
///
/// Every value in a `WWW-Authenticate` auth-param here is config-derived, so this
/// is defence against a typo in `config.yaml` producing a header that a client
/// parses as something other than intended — not against an attacker. Cheaper than
/// validating URLs at load time and it keeps the header well-formed regardless.
fn quoted(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Derive a resource's protected-resource metadata URL, per RFC 9728 §3: the
/// well-known segment goes between the authority and the resource's path, NOT at
/// the end. For `https://kb.example.com/mcp` that is
/// `https://kb.example.com/.well-known/oauth-protected-resource/mcp` — which is
/// also why clients probe the path-suffixed form before the bare one, and why
/// `server::assemble_router` registers both.
fn resource_metadata_url(resource: &str) -> String {
    let trimmed = resource.trim();
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        // Not a URL we can take apart. `Config::resolve` already refuses an empty
        // `resource`, so this is a malformed non-empty value; append rather than
        // panic, so a bad config surfaces as a discovery 404 with the offending
        // string visible in the metadata document, not as a crash at startup.
        return format!(
            "{}{PROTECTED_RESOURCE_METADATA_PREFIX}",
            trimmed.trim_end_matches('/')
        );
    };
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].trim_end_matches('/')),
        None => (rest, ""),
    };
    format!("{scheme}://{authority}{PROTECTED_RESOURCE_METADATA_PREFIX}{path}")
}

#[cfg(test)]
pub(crate) mod testing {
    //! Shared JWT/JWKS fixtures. `pub(crate)` because `server.rs`'s middleware and
    //! router tests need to mint the same tokens this module's own tests do, and a
    //! second copy of a throwaway keypair in another file is a copy that drifts.

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Throwaway 2048-bit RSA keypair, generated for this test suite and used
    /// nowhere else. `KEY_B` exists only to produce a signature that `KEY_A`'s
    /// public half must reject.
    pub const KID_A: &str = "test-key-a";
    pub const N_A: &str = "zXtrd9E8iuVecx_7KN0nxRV0m0DgZayGgW5D4bPJMwUcFX6SIsyYpSCAGjT1Fia85xH-YrMxk9XSjuMpYB8GphQ5NitAaVx8CQeoVQw8WEi1YSG53OfuSftmkX79D48nVP6VxKq3JW_RIaTM8xsisVV2zzFeQVN_NsFNCAsClYoXLUj8Wfc9WsFz8DszbQep6I4gceD6WNCs72AQMXR5vIOfGxK5eP5JWOjK7FN95njVNbXY6p5QUQii_3HkFSDQv9drzpzeKXdDziFdSG5qZfMwGuqjfCMDNfwYKxC4AbAGbtSTCHFEWe0CuWX95xgqvyJCsVjkh8xMz-WpPoWLSQ";

    pub const KEY_A_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDNe2t30TyK5V5z
H/so3SfFFXSbQOBlrIaBbkPhs8kzBRwVfpIizJilIIAaNPUWJrznEf5iszGT1dKO
4ylgHwamFDk2K0BpXHwJB6hVDDxYSLVhIbnc5+5J+2aRfv0PjydU/pXEqrclb9Eh
pMzzGyKxVXbPMV5BU382wU0ICwKVihctSPxZ9z1awXPwOzNtB6nojiBx4PpY0Kzv
YBAxdHm8g58bErl4/klY6MrsU33meNU1tdjqnlBRCKL/ceQVINC/12vOnN4pd0PO
IV1Ibmpl8zAa6qN8IwM1/BgrELgBsAZu1JMIcURZ7QK5Zf3nGCq/IkKxWOSHzEzP
5ak+hYtJAgMBAAECggEAPpyQWxKZGZOZi4ffroxw3VdT0CjdF24SECdKrN/s+0xf
ydbm7Y6dJpe4IQQo+AZ2wgwUEPwcK7lYLuzeAymBC6MW6cAVIOWq789zBfM0Agyp
o/60VTEgxU9C6iuhLZgHupjWhvYj11byiQdf4eXPVOy/RpP67fnkxgjxkXVVZL4C
zJ5KQZRLi+DH9l5Vd5nKqyRVVFVaaD0ws5Lw7n2HBrraq/omV6FlcIkePB4Tx2gD
WudBhUPnrhukXaoEWEvBNXnVSExU+bZMeWvQdcGVL6OE1LG9IqsjYiumF2kb0n0L
ZTalPbtAHoNDEKIG2+rwCqsLBZQvFnLcFTlc1WAExwKBgQDr8IjWaZ8I0mQHbRhM
BsrLDjf2qBVclBMchYafmh6NoI5E2+928NTo/uxssGTEX4ce0v6CW6dHo8vNVfN/
cUxjQW479qvugi8EBp6rQ9ZOjSra078L145jTCaJLfxMuYDgjycvcv8wOqxJL/80
F1Qjkn+pGsQFvjCAEikjwcG0cwKBgQDe8/J8W2tPpnRiv40T2Hgu4Th8gRfm/79k
RBZqeiO/EiI9zwHmOj9s02fK3tBPyQgZMyQyQNMJjFuWzCjHE8gaBdLAqGrpILL2
jR7EvXBPrGRpWcbnRCyODURcaIY1dTVImT1g9rhwzDJNQFd7XPGR60LCMUxL8b2p
hlgFlw9OUwKBgFAqvpP792mL8ykCzIqolCdCgYlxuzBlr8i1JfT87Py6XRzQjiEf
23f/hl234cVHoCW9E3U/pysUYJ84YTAgUxA2nzoIqoqz+T2o8ijHN/4gwTrxT6y6
ZUsgCMf7tAptzXh/q5TXwhWlGf0ULeaJNrGPiYjv60L4SIp7oTbhEuw5AoGAdJg8
yn4Am7HgEbg87hD5oQKVSL82Ic7DZ4sX8e0X/pdcIti8FIuHmcDg+b4WUHNAcfVF
y6YM92RYjX8NIDcfIUTEV458ApjgHoHkglzTfEcaZ+HUXCNR7aPQiUb8UL6P8/x3
ldrQz+Rpte6dEV2k03ul+OpRDTJJznr8U0gRcBMCgYB/aUF16/RJvW5nLGWTbAS8
D4d9SgETq0P0zbuDUk60Fk6kQbQ+bwX+ffgsEP/P/CFTNJ+opoCo0/6uK8WlKs15
uVx1QLn2oATcEUusHESeflBUSSaYlhHXFL7ahvAgBs3vzgWZnUVnz2A3QDCiLu6H
EmjUKNFGC0zInLUM1Cbu9w==
-----END PRIVATE KEY-----
";

    pub const KEY_B_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQClweugXTF1SY0q
ar8Z68ong9eCzOI3kCSipuiCDhVPad8Gn4be0RM4B7t342iuG4UjyXnCQpCoWGiN
L4KN52hFBE7M8c/7JutAtmpJFm33cFKZ+yWfAcX5FFtC/BdOPfaPtije98QJRmlv
lJ6n7c8uMpXhtV1ZIqwm9g7chVWlUHKAgMFGaUeKWdksQ9tTZgDKeHO1vfRZZlYT
XdvDpNe7Dxz1o3eefTsrKsE1DDTXrDfJPUDPPpBTMmT+xrPRehuNqrNQRUJWEIAR
bJpV5ltnhNX4zs3YQ39/XTCcQjnbu4wRpDUgTIhPuomg18t8vqi1CbhaN8+Ww3oU
FRdXKE59AgMBAAECggEAMeZ2umDD3mTFmCLpo/KNeabhrrFiWsrMlKC9t0VpGe6r
4xEMZ7C2YfRF9hoibePACZ2CR76FUQDIfNR0L6ceB0T8OguECr5VLTadOaKEeWy5
mTx3v24nvMvpi3lbxMS3oNz8Yd9iB07It/wcZT6c0/ILmBbi4s4i2FnT8IQ9W9Ym
iuMwqujeyrEUG/O3HrUJLHNe6PwJj6s8mbAKxfmCqnDLCyWlQejR2FL24mriCSQD
G03gZ6VazAnDt19SjToPKH1e6XjB6FySUX3gDhA9yXSPphCdaa9Ov9w/4UktlEtz
RRobV9e+e5e2qUrv77CZu3PMlH/gZARM/ncSCKlYGwKBgQDqVHUYvBArQWLO2po6
9SvjcvB631+cUO94k+nV4vlbXbzjMznXPPShijf5cirzRhazSRqxp5Yt6959vVDI
Pe/vjP0lM0dLZwiOnsEaq+ArEoAnidD09bUn81qMnsH3eUpXtIthB5ltkNX3tXb2
Xs04OxDm4IrSoMg7/w2HXb4YbwKBgQC1FhMz8PvXDAb7uJ5ydjd4Gw1CEqw93YSf
S1koX2x86qhELfUglAHb+h5RHxE7zqi5fqzrsHl3Ow3392O3clcqrbME4u//fvmV
XCr7eHraeIByX/ZpnBiiuYjrvN6MKDUy00yBDdGMGGA0JT2+aH/06qg/G5xG8SiV
0ajx8wAF0wKBgQDnbTQckpfRcIk6TBFoSvzmbHzujS9rPU/UoRifAcRNpP1I0i28
0lm0NMLlXAjpLH584KU5cY7TmZCqVE+1A960kmTs2YD/CiocWNPUGI2TXHkvE2BI
nWYlp6T1HlHorGRszEWfNZck65c2RoTP+37omwUtT/Qq41n+Tv44g6+bhwKBgGyQ
EW8gWDsycLVUl1lT2ildPnOQMkbcmPfO+mKj4qx5GevWCZFAamTw7GAB2hka6jha
41xhblC2zMcOP2/pUqy5egvB6dQo0YRjvzkHn89+UrM/KMFj3bkgth9uGZW5PTt9
Re5Q1IHC01ovwXZ3u86fJ8K90NEPHx/ClCCJaEgVAoGAHlopDQ/w7JN5sCBYDZeE
eAfND1Q/hnbfjdUgg13/Qmhqwm86RYJ3E9mxjcCNKZ3hNX3Xcs5NW5oC9tj9Nb9G
B5bK2earcA3sKw66Uvzd5AtypET7/RPOSgpXOD34f1RN38fWqc+L0pdHZ41D5eif
13kE7LEf//HMi5ix93dRdZw=
-----END PRIVATE KEY-----
";

    pub const ISSUER: &str = "https://authentik.example.test/application/o/mcp-kb-rag/";
    pub const AUDIENCE: &str = "test-client-id";
    pub const RESOURCE: &str = "https://kb.example.test/mcp";

    /// A JWK Set carrying only `KEY_A`'s public half, as an AS would serve it.
    pub fn jwks_body() -> String {
        serde_json::json!({
            "keys": [{
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": KID_A,
                "n": N_A,
                "e": "AQAB",
            }]
        })
        .to_string()
    }

    /// Seconds since the epoch, for `exp`.
    pub fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Mint a token with full control over every field a test might want wrong.
    /// `pem` is which key signs it; `kid` is what the header *claims* signed it —
    /// letting a test say "signed by B, labelled A" for the bad-signature case.
    pub fn mint(pem: &str, kid: &str, claims: serde_json::Value) -> String {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(kid.to_string());
        encode(
            &header,
            &claims,
            &EncodingKey::from_rsa_pem(pem.as_bytes()).unwrap(),
        )
        .unwrap()
    }

    /// The happy-path token: right key, right issuer, right audience, valid for an
    /// hour, carrying `mcp:read mcp:write`.
    pub fn valid_token() -> String {
        mint(
            KEY_A_PEM,
            KID_A,
            serde_json::json!({
                "iss": ISSUER,
                "aud": AUDIENCE,
                "azp": AUDIENCE,
                "sub": "user-1",
                "exp": now() + 3600,
                "scope": "mcp:read mcp:write",
            }),
        )
    }

    /// A throwaway loopback HTTP server that answers every request with one canned
    /// response, counting hits. Hand-rolled for the same reason `rerank.rs`'s
    /// `FakeReranker` is: the repo has no HTTP-mock dev-dependency and this is
    /// cheaper than adding one.
    pub struct FakeJwksServer {
        pub url: String,
        pub hits: Arc<AtomicUsize>,
    }

    pub async fn spawn_jwks_server(status_line: &'static str, body: String) -> FakeJwksServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&hits);

        tokio::spawn(async move {
            let body = Arc::new(body);
            while let Ok((mut sock, _)) = listener.accept().await {
                let counter = Arc::clone(&counter);
                let body = Arc::clone(&body);
                tokio::spawn(async move {
                    // A GET has no body, so end-of-headers is end-of-request.
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 4096];
                    loop {
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    counter.fetch_add(1, Ordering::SeqCst);
                    let resp = format!(
                        "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                    let _ = sock.flush().await;
                });
            }
        });

        FakeJwksServer {
            url: format!("http://{addr}/jwks"),
            hits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use std::sync::atomic::Ordering;

    fn oauth_config(jwks_uri: &str) -> ResolvedOAuthConfig {
        ResolvedOAuthConfig {
            issuer: ISSUER.to_string(),
            jwks_uri: jwks_uri.to_string(),
            audience: AUDIENCE.to_string(),
            resource: RESOURCE.to_string(),
            required_scope: "mcp:read".to_string(),
            scopes_supported: vec!["mcp:read".to_string(), "mcp:write".to_string()],
        }
    }

    /// Zero cooldown: a test that wants to observe a refetch should not have to
    /// sleep out `JWKS_MIN_REFETCH_INTERVAL`.
    fn validator_no_cooldown(jwks_uri: &str) -> OAuthValidator {
        OAuthValidator::build(&oauth_config(jwks_uri), Duration::ZERO).unwrap()
    }

    fn validator(jwks_uri: &str) -> OAuthValidator {
        OAuthValidator::new(&oauth_config(jwks_uri)).unwrap()
    }

    // ── metadata URL derivation (RFC 9728 §3) ────────────────────────────────

    #[test]
    fn metadata_url_splices_the_well_known_segment_before_the_path() {
        // The well-known segment goes between authority and path, NOT appended.
        assert_eq!(
            resource_metadata_url("https://kb.example.com/mcp"),
            "https://kb.example.com/.well-known/oauth-protected-resource/mcp"
        );
    }

    #[test]
    fn metadata_url_for_a_path_less_resource_is_the_bare_well_known() {
        assert_eq!(
            resource_metadata_url("https://kb.example.com"),
            "https://kb.example.com/.well-known/oauth-protected-resource"
        );
        assert_eq!(
            resource_metadata_url("https://kb.example.com/"),
            "https://kb.example.com/.well-known/oauth-protected-resource"
        );
    }

    #[test]
    fn metadata_url_keeps_a_port_and_drops_a_trailing_slash() {
        assert_eq!(
            resource_metadata_url("http://localhost:8001/mcp/"),
            "http://localhost:8001/.well-known/oauth-protected-resource/mcp"
        );
    }

    #[test]
    fn metadata_url_of_a_malformed_resource_does_not_panic() {
        // `Config::resolve` refuses an empty `resource`, so this is a non-empty but
        // scheme-less value — a config typo, which must degrade to a 404 at
        // discovery rather than a startup crash.
        assert_eq!(
            resource_metadata_url("kb.example.com/mcp"),
            "kb.example.com/mcp/.well-known/oauth-protected-resource"
        );
    }

    // ── the metadata document and the challenge headers ──────────────────────

    #[test]
    fn metadata_document_has_the_rfc_9728_shape() {
        let v = validator("http://127.0.0.1:1/jwks");
        let doc = v.metadata();
        assert_eq!(doc["resource"], RESOURCE);
        // Byte-identical, trailing slash and all — a client matches this against
        // the `iss` of the tokens it receives.
        assert_eq!(doc["authorization_servers"][0], ISSUER);
        assert_eq!(doc["scopes_supported"][0], "mcp:read");
        assert_eq!(doc["scopes_supported"][1], "mcp:write");
        assert_eq!(doc["bearer_methods_supported"][0], "header");
        assert!(doc["resource_name"].is_string());
    }

    #[test]
    fn invalid_token_challenge_is_well_formed() {
        let v = validator("http://127.0.0.1:1/jwks");
        assert_eq!(
            v.invalid_token_challenge(),
            "Bearer error=\"invalid_token\", \
             resource_metadata=\"https://kb.example.test/.well-known/oauth-protected-resource/mcp\", \
             scope=\"mcp:read mcp:write\""
        );
    }

    #[test]
    fn insufficient_scope_challenge_names_the_missing_scope_not_the_menu() {
        let v = validator("http://127.0.0.1:1/jwks");
        assert_eq!(
            v.insufficient_scope_challenge(),
            "Bearer error=\"insufficient_scope\", scope=\"mcp:read\", \
             resource_metadata=\"https://kb.example.test/.well-known/oauth-protected-resource/mcp\""
        );
    }

    #[test]
    fn challenge_values_are_escaped_not_pasted() {
        assert_eq!(quoted(r#"a"b\c"#), r#"a\"b\\c"#);
    }

    // ── token validation ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn a_well_formed_token_is_accepted_and_yields_its_scopes() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let token = v.validate(&valid_token()).await.unwrap();
        assert_eq!(token.subject.as_deref(), Some("user-1"));
        assert_eq!(token.scopes, vec!["mcp:read", "mcp:write"]);
        assert!(token.has_scope("mcp:write"));
    }

    #[tokio::test]
    async fn aud_is_accepted_as_a_string_and_as_an_array() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);

        for aud in [
            serde_json::json!(AUDIENCE),
            serde_json::json!(["some-other-client", AUDIENCE]),
        ] {
            let token = mint(
                KEY_A_PEM,
                KID_A,
                serde_json::json!({
                    "iss": ISSUER, "aud": aud, "exp": now() + 3600, "scope": "mcp:read",
                }),
            );
            assert!(
                v.validate(&token).await.is_ok(),
                "aud must be accepted in both RFC 7519 §4.1.3 shapes"
            );
        }
    }

    #[tokio::test]
    async fn a_wrong_issuer_is_rejected() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        // Same issuer minus the trailing slash: the near-miss that actually happens
        // in practice, not an obviously foreign string.
        let token = mint(
            KEY_A_PEM,
            KID_A,
            serde_json::json!({
                "iss": ISSUER.trim_end_matches('/'),
                "aud": AUDIENCE, "exp": now() + 3600, "scope": "mcp:read",
            }),
        );
        assert!(matches!(
            v.validate(&token).await,
            Err(TokenRejection::Invalid(_))
        ));
    }

    #[tokio::test]
    async fn a_missing_issuer_or_audience_is_rejected() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        // jsonwebtoken only checks iss/aud when the claim is present, so omitting
        // them entirely is the way a token would sneak past a validator that had
        // not set `required_spec_claims`.
        for claims in [
            serde_json::json!({"aud": AUDIENCE, "exp": now() + 3600, "scope": "mcp:read"}),
            serde_json::json!({"iss": ISSUER, "exp": now() + 3600, "scope": "mcp:read"}),
        ] {
            let token = mint(KEY_A_PEM, KID_A, claims);
            assert!(matches!(
                v.validate(&token).await,
                Err(TokenRejection::Invalid(_))
            ));
        }
    }

    #[tokio::test]
    async fn a_wrong_audience_is_rejected() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let token = mint(
            KEY_A_PEM,
            KID_A,
            serde_json::json!({
                "iss": ISSUER, "aud": "some-other-client",
                "exp": now() + 3600, "scope": "mcp:read",
            }),
        );
        assert!(matches!(
            v.validate(&token).await,
            Err(TokenRejection::Invalid(_))
        ));
    }

    #[tokio::test]
    async fn an_expired_token_is_rejected_beyond_the_leeway() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let token = mint(
            KEY_A_PEM,
            KID_A,
            serde_json::json!({
                "iss": ISSUER, "aud": AUDIENCE,
                "exp": now() - (CLOCK_SKEW_LEEWAY_SECS + 60),
                "scope": "mcp:read",
            }),
        );
        assert!(matches!(
            v.validate(&token).await,
            Err(TokenRejection::Invalid(_))
        ));
    }

    #[tokio::test]
    async fn a_token_signed_by_the_wrong_key_is_rejected() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        // Signed by B but LABELLED as A, so the lookup succeeds and the failure is
        // genuinely a signature failure rather than an unknown-kid failure.
        let token = mint(
            KEY_B_PEM,
            KID_A,
            serde_json::json!({
                "iss": ISSUER, "aud": AUDIENCE, "exp": now() + 3600, "scope": "mcp:read",
            }),
        );
        assert!(matches!(
            v.validate(&token).await,
            Err(TokenRejection::Invalid(_))
        ));
    }

    #[tokio::test]
    async fn a_valid_token_without_the_required_scope_is_insufficient_not_invalid() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        for scope in [serde_json::json!("openid profile"), serde_json::json!("")] {
            let token = mint(
                KEY_A_PEM,
                KID_A,
                serde_json::json!({
                    "iss": ISSUER, "aud": AUDIENCE, "exp": now() + 3600, "scope": scope,
                }),
            );
            assert_eq!(
                v.validate(&token).await.unwrap_err(),
                TokenRejection::InsufficientScope,
                "the token itself is fine — conflating this with invalid_token sends \
                 the client round the authorization flow to the same refusal"
            );
        }
    }

    #[tokio::test]
    async fn a_token_with_no_scope_claim_at_all_is_insufficient() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let token = mint(
            KEY_A_PEM,
            KID_A,
            serde_json::json!({"iss": ISSUER, "aud": AUDIENCE, "exp": now() + 3600}),
        );
        assert_eq!(
            v.validate(&token).await.unwrap_err(),
            TokenRejection::InsufficientScope
        );
    }

    #[tokio::test]
    async fn a_non_rs256_token_is_rejected_before_any_jwks_fetch() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(Algorithm::HS256),
            &serde_json::json!({"iss": ISSUER, "aud": AUDIENCE, "exp": now() + 3600}),
            &jsonwebtoken::EncodingKey::from_secret(b"not-the-signing-key"),
        )
        .unwrap();
        assert!(matches!(
            v.validate(&token).await,
            Err(TokenRejection::Invalid(_))
        ));
        assert_eq!(
            jwks.hits.load(Ordering::SeqCst),
            0,
            "a junk algorithm must not be able to schedule IdP traffic"
        );
    }

    #[tokio::test]
    async fn garbage_in_the_authorization_header_is_rejected() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        for junk in ["", "not-a-jwt", "a.b.c"] {
            assert!(matches!(
                v.validate(junk).await,
                Err(TokenRejection::Invalid(_))
            ));
        }
    }

    // ── JWKS fetching ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn the_jwks_is_fetched_once_and_cached() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        for _ in 0..3 {
            v.validate(&valid_token()).await.unwrap();
        }
        assert_eq!(
            jwks.hits.load(Ordering::SeqCst),
            1,
            "a cached key must not be re-fetched per request"
        );
    }

    #[tokio::test]
    async fn an_unknown_kid_does_not_refetch_during_the_cooldown() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url); // real 60s cooldown
        // First call populates the cache (one fetch); the unknown kid is then NOT
        // worth a second fetch, because we just fetched.
        let token = mint(
            KEY_A_PEM,
            "rotated-key",
            serde_json::json!({
                "iss": ISSUER, "aud": AUDIENCE, "exp": now() + 3600, "scope": "mcp:read",
            }),
        );
        for _ in 0..5 {
            assert!(matches!(
                v.validate(&token).await,
                Err(TokenRejection::Invalid(_))
            ));
        }
        assert_eq!(
            jwks.hits.load(Ordering::SeqCst),
            1,
            "kid is attacker-controlled — five junk tokens must not mean five IdP hits"
        );
    }

    #[tokio::test]
    async fn an_unknown_kid_refetches_once_the_cooldown_has_passed() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator_no_cooldown(&jwks.url);
        let token = mint(
            KEY_A_PEM,
            "rotated-key",
            serde_json::json!({
                "iss": ISSUER, "aud": AUDIENCE, "exp": now() + 3600, "scope": "mcp:read",
            }),
        );
        assert!(matches!(
            v.validate(&token).await,
            Err(TokenRejection::Invalid(_))
        ));
        assert!(matches!(
            v.validate(&token).await,
            Err(TokenRejection::Invalid(_))
        ));
        assert_eq!(
            jwks.hits.load(Ordering::SeqCst),
            2,
            "with the cooldown elapsed, an unknown kid must trigger a refresh — this \
             is how a rotated signing key is picked up without a restart"
        );
    }

    #[tokio::test]
    async fn an_unreachable_jwks_endpoint_fails_closed() {
        // Port 1 refuses instantly — the same unreachable-backend trick the status
        // and rerank tests use.
        let v = validator("http://127.0.0.1:1/jwks");
        assert!(
            matches!(
                v.validate(&valid_token()).await,
                Err(TokenRejection::Invalid(_))
            ),
            "an IdP we cannot reach must mean 'no', never 'sure'"
        );
    }

    #[tokio::test]
    async fn a_jwks_error_response_fails_closed() {
        let jwks = spawn_jwks_server("500 Internal Server Error", "{}".into()).await;
        let v = validator(&jwks.url);
        assert!(matches!(
            v.validate(&valid_token()).await,
            Err(TokenRejection::Invalid(_))
        ));
    }

    #[tokio::test]
    async fn a_key_set_with_no_usable_keys_fails_closed() {
        let body = serde_json::json!({
            // An EC key and an RSA key with no `kid`: both unusable here, and the
            // set must not be accepted as "fetched successfully, zero keys".
            "keys": [
                {"kty": "EC", "crv": "P-256", "kid": "ec", "x": "AAAA", "y": "AAAA"},
                {"kty": "RSA", "n": N_A, "e": "AQAB"},
            ]
        })
        .to_string();
        let jwks = spawn_jwks_server("200 OK", body).await;
        let v = validator(&jwks.url);
        assert!(matches!(
            v.validate(&valid_token()).await,
            Err(TokenRejection::Invalid(_))
        ));
    }

    #[tokio::test]
    async fn a_kid_less_header_uses_the_single_published_key() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let token = jsonwebtoken::encode(
            // No `kid` on the header at all.
            &jsonwebtoken::Header::new(Algorithm::RS256),
            &serde_json::json!({
                "iss": ISSUER, "aud": AUDIENCE, "exp": now() + 3600, "scope": "mcp:read",
            }),
            &jsonwebtoken::EncodingKey::from_rsa_pem(KEY_A_PEM.as_bytes()).unwrap(),
        )
        .unwrap();
        assert!(v.validate(&token).await.is_ok());
    }
}
