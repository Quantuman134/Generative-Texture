import torch

class MultiDirectionalLights():
    def __init__(
            self,
            intensities=[1.0],
            directions=[[1.0, 1.0, 1.0]],
            device='cpu'
    ) -> None:
        self.num = len(intensities)
        self.lights = []
        for i in range(self.num):
            intensity = torch.tensor(intensities[i], dtype=torch.float32, device=device)
            direction = torch.tensor(directions[i], dtype=torch.float32, device=device)
            self.lights.append(CustomDirectionalLight(intensity, direction))

class CustomDirectionalLight():
    def __init__(
        self,
        intensity,
        direction
    ):
        self.intensity = intensity
        self.direction = direction