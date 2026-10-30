import pygame

from engines.base_engine import BACKGROUND_COLOR, BaseEngine


class PygameEngine(BaseEngine):
    def setup(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def update(self, audio_features: dict, delta_time: float) -> None:
        pass

    def render(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)
        pygame.display.flip()

    def teardown(self) -> None:
        pygame.quit()
