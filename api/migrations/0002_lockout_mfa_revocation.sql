-- Bloqueio por conta + ORIGEM (IPv4, ou prefixo /64 do IPv6). Antes, 5 senhas erradas de
-- qualquer lugar trancavam a conta: quem soubesse o e-mail mantinha a vítima de fora.
-- Agora cada origem tem o seu contador; o bloqueio da conta inteira só acontece com volume alto
-- (ataque distribuído) e não vale para origens onde o usuário já entrou com sucesso.
CREATE TABLE login_attempts (
    user_id         uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    origin          text NOT NULL CHECK (char_length(origin) BETWEEN 1 AND 64),
    failed_attempts integer NOT NULL DEFAULT 0 CHECK (failed_attempts >= 0),
    locked_until    timestamptz,
    last_failure_at timestamptz,
    last_success_at timestamptz,
    PRIMARY KEY (user_id, origin)
);
CREATE INDEX login_attempts_failure_idx ON login_attempts (last_failure_at);

ALTER TABLE users
    -- Janela do contador da conta inteira (zera após 24 h sem falhas).
    ADD COLUMN last_failed_at     timestamptz,
    -- MFA (TOTP): segredo cifrado (ChaCha20-Poly1305, amarrado ao id do usuário) e o último
    -- passo de 30 s aceito (impede reaproveitar o mesmo código).
    ADD COLUMN totp_secret_enc    bytea,
    ADD COLUMN totp_last_step     bigint,
    -- Access tokens (JWT) emitidos até este instante deixam de valer.
    ADD COLUMN tokens_valid_after timestamptz;
