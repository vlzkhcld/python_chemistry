class FollowGradient:
    def __init__(self, speed=1.):
        self.speed = speed

    def __call__(self, grad, **kwargs):
        return -self.speed * grad

