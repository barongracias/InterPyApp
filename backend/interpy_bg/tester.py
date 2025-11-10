
from interpy_bg.logger import get_console_logger
logger = get_console_logger(__name__)


class Trainer(NeuralNetwork):
    def __init__(self, input_size):
        super().__init__(input_size)
        self.input_size = 10