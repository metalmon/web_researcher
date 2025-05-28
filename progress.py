from tqdm import tqdm
from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager

class PipelineProgress:
    """
    Utility class for managing a single progress bar across different pipeline stages.
    Provides a clean interface for updating progress and stage descriptions.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PipelineProgress, cls).__new__(cls)
        return cls._instance

    def __init__(self, initial_desc: str = "Инициализация"):
        if not self._initialized:
            self.pbar: Optional[tqdm] = None
            self.initial_desc = initial_desc
            self.current_stage: Dict[str, Any] = {
                'description': initial_desc,
                'total': 100,
                'current': 0,
                'substage': None
            }
            self.total_steps = 0
            self.current_step = 0
            self._initialized = True

    def write(self, message):
        """Выводит сообщение над прогресс-баром."""
        if self.pbar:
            self.pbar.write(message)
        else:
            print(message)

    def _create_progress_bar(self):
        """Create a new progress bar with initial settings."""
        if self.pbar is not None:
            self.close()
        print("\033[K", end="")  # Очищаем текущую строку
        self.pbar = tqdm(
            total=self.total_steps,
            desc=self.initial_desc,
            leave=False,
            position=0,
            bar_format='{desc:40}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        self.current_step = 0

    def update_stage(self, description: str, total: int, substage: Optional[str] = None):
        """
        Update the progress bar for a new stage.
        
        Args:
            description: New stage description
            total: Total number of steps in this stage
            substage: Optional substage description (e.g., "Страница 1/4, Проход 2/4")
        """
        if self.pbar is None:
            self.total_steps = total
            self._create_progress_bar()
        else:
            # Обновляем общее количество шагов
            self.total_steps = total
            self.pbar.total = total
        
        self.current_stage = {
            'description': description,
            'total': total,
            'current': 0,
            'substage': substage
        }
        
        # Формируем полное описание с учетом подэтапа
        full_desc = description
        if substage and substage != description:  # Добавляем подэтап только если он отличается от основного описания
            full_desc = f"{description} ({substage})"
        
        # Обновляем прогресс-бар без сброса
        self.pbar.set_description(full_desc)

    def update(self, n: int = 1, substage: Optional[str] = None):
        """
        Update progress by n steps.
        
        Args:
            n: Number of steps to update
            substage: Optional substage description to update
        """
        if self.pbar is None:
            self._create_progress_bar()
            
        if substage and substage != self.current_stage['substage']:
            self.current_stage['substage'] = substage
            full_desc = f"{self.current_stage['description']} ({substage})"
            self.pbar.set_description(full_desc)
        self.pbar.update(n)
        self.current_stage['current'] += n
        self.current_step += n

    def close(self):
        """Close the progress bar."""
        if self.pbar is not None:
            self.pbar.clear()  # Очищаем вывод
            self.pbar.close()  # Закрываем прогресс-бар
            print("\033[K", end="")  # Очищаем текущую строку
            self.pbar = None

    @contextmanager
    def stage(self, description: str, total: int):
        """
        Context manager for a pipeline stage.
        
        Example:
            with progress.stage("Скрапинг страниц", len(links)):
                for link in links:
                    process_link(link)
                    progress.update(1)
        """
        self.update_stage(description, total)
        try:
            yield
        finally:
            self.close()

    def __del__(self):
        """Ensure progress bar is closed when object is destroyed."""
        self.close()

# Global progress bar instance
progress = PipelineProgress()

def with_progress(description: str, total: int):
    """
    Decorator for running a function with progress bar updates.
    
    Example:
        @with_progress("Обработка данных", 10)
        def process_data():
            for i in range(10):
                do_something()
                progress.update(1)
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            progress.update_stage(description, total)
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator 