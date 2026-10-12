proxy_pass http://api_ocr_upstream;
proxy_http_version 1.1;
proxy_set_header Connection "";
proxy_set_header Host $host;
# SOBRESCREVE (não acrescenta): o cliente não consegue forjar a cadeia de X-Forwarded-For.
proxy_set_header X-Forwarded-For   $remote_addr;
proxy_set_header X-Forwarded-Proto https;
proxy_connect_timeout 3s;
proxy_next_upstream off;
# O HSTS e os demais cabeçalhos já vêm do nginx (evita duplicar os do app).
proxy_hide_header Strict-Transport-Security;
proxy_hide_header X-Content-Type-Options;
proxy_hide_header X-Frame-Options;
proxy_hide_header Referrer-Policy;
