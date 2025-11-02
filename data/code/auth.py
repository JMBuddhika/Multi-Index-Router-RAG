def validate_jwt(token: str) -> bool:
    """Toy example for demo purposes only."""
    return token.startswith("eyJ")
