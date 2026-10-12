-- Usuários. E-mail sempre normalizado (minúsculas) pela aplicação; o CHECK garante.
CREATE TABLE users (
    id              uuid PRIMARY KEY,
    email           text NOT NULL UNIQUE
                    CHECK (email = lower(email) AND char_length(email) BETWEEN 3 AND 254),
    password_hash   text NOT NULL,
    failed_attempts integer NOT NULL DEFAULT 0 CHECK (failed_attempts >= 0),
    locked_until    timestamptz,
    is_active       boolean NOT NULL DEFAULT true,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Clientes OAuth2 (públicos: autenticam-se via PKCE, sem segredo).
CREATE TABLE oauth_clients (
    client_id       text PRIMARY KEY CHECK (client_id ~ '^[A-Za-z0-9._-]{3,64}$'),
    name            text NOT NULL CHECK (char_length(name) BETWEEN 1 AND 100),
    redirect_uris   text[] NOT NULL CHECK (cardinality(redirect_uris) >= 1),
    allowed_scopes  text[] NOT NULL CHECK (cardinality(allowed_scopes) >= 1),
    is_active       boolean NOT NULL DEFAULT true,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Authorization codes: só o hash SHA-256 é guardado; uso único; TTL curto.
CREATE TABLE auth_codes (
    code_hash       bytea PRIMARY KEY CHECK (octet_length(code_hash) = 32),
    client_id       text NOT NULL REFERENCES oauth_clients(client_id) ON DELETE CASCADE,
    user_id         uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    redirect_uri    text NOT NULL,
    scope           text NOT NULL,
    code_challenge  text NOT NULL,
    expires_at      timestamptz NOT NULL,
    used_at         timestamptz
);
CREATE INDEX auth_codes_expires_idx ON auth_codes (expires_at);

-- Refresh tokens com rotação. `family_id` agrupa a cadeia de rotações;
-- reutilizar um token já usado revoga a família inteira.
CREATE TABLE refresh_tokens (
    id                uuid PRIMARY KEY,
    token_hash        bytea NOT NULL UNIQUE CHECK (octet_length(token_hash) = 32),
    family_id         uuid NOT NULL,
    family_started_at timestamptz NOT NULL,
    user_id           uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    client_id         text NOT NULL REFERENCES oauth_clients(client_id) ON DELETE CASCADE,
    scope             text NOT NULL,
    expires_at        timestamptz NOT NULL,
    used_at           timestamptz,
    revoked_at        timestamptz,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX refresh_tokens_family_idx ON refresh_tokens (family_id);
CREATE INDEX refresh_tokens_user_client_idx ON refresh_tokens (user_id, client_id);
CREATE INDEX refresh_tokens_expires_idx ON refresh_tokens (expires_at);

-- Trilha de auditoria de eventos de segurança (sem segredos).
CREATE TABLE audit_log (
    id        bigserial PRIMARY KEY,
    at        timestamptz NOT NULL DEFAULT now(),
    kind      text NOT NULL,
    user_id   uuid,
    client_id text,
    ip        text,
    detail    text
);
CREATE INDEX audit_log_at_idx ON audit_log (at);
