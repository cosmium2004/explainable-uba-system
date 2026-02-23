import secrets
print(secrets.token_hex(32))  # For SECRET_KEY
print(secrets.token_hex(16))  # For API_KEY