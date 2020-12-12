import time
from datetime import datetime


class Budget:

    def __init__(self):
        self.modulo_counter_generations = 0
        self.population_size = None

    def add(self, budget_type: str, amount: int):
        self.budget_type = budget_type
        self.amount = amount
        if budget_type == "time_in_h":
            self.start_time = time.time()

    def is_not_used_up(self) -> bool:
        if self.budget_type == 'time_in_h':
            is_used_up = self._is_time_budget_left()
        elif self.budget_type == 'runs':
            is_used_up = self._is_runs_budget_left()
        elif self.budget_type == 'generations':
            is_used_up = self._is_generations_budget_left()
        else:
            is_used_up = True
        self.modulo_counter_generations += 1
        return is_used_up

    def _is_time_budget_left(self) -> bool:
        time_passed_since_start = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(self.start_time)
        if (time_passed_since_start.total_seconds()) / 3600 < self.amount:
            return True
        else:
            return False

    def _is_runs_budget_left(self) -> bool:
        if self.amount > 0:
            self.amount -= 1
            return True
        else:
            return False

    def _is_generations_budget_left(self) -> bool:
        if self._is_generation_over():
            self.amount -= 1
            return True
        elif self._is_last_generation():
            self.amount -= 1
            return False

        elif self.amount > 0:
            return True

    def _is_generation_over(self) -> bool:
        # Decrease generations every X iterations (X is amount of individuals per generation)
        if self.amount > 1 \
                and self.modulo_counter_generations % self.population_size == 0 \
                and self.modulo_counter_generations != 0:
            return True
        else:
            return False

    def _is_last_generation(self) -> bool:
        # If last generation, budget is used up
        if self.amount == 1 \
                and self.modulo_counter_generations % self.population_size == 0 \
                and self.modulo_counter_generations != 0:
            return True
        else:
            return False

    def set_population_size(self, population_size: int):
        self.population_size = population_size
