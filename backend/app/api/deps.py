from fastapi import Request


def get_optimizer(request: Request):
    """Return the optimizer loaded once by the application lifespan."""
    return request.app.state.optimizer
