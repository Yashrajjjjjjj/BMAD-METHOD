"""BMAD Agent - AI-Driven Orchestration Module

This module provides a starter scaffold for a BMAD-compliant agent that can
handle AI-driven orchestration tasks. It includes initialization, async task
handling, and core methods for interfacing with the BMAD ecosystem.

Purpose:
    Enable AI-driven orchestration and automation within the BMAD methodology
    framework, providing a foundation for intelligent agent-based control
    systems that can monitor, analyze, and orchestrate complex workflows.

Compliance:
    This agent scaffold follows BMAD (Better Methods for Accelerated Development)
    principles, emphasizing modularity, async operations, and extensibility.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for BMAD Agent."""
    name: str = "BMADAgent"
    version: str = "0.1.0"
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # seconds
    retry_attempts: int = 3
    enable_logging: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Represents an asynchronous task for the agent."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None


class BMADAgent:
    """BMAD Agent for AI-Driven Orchestration.
    
    This class provides a foundation for building intelligent agents that can
    orchestrate and automate tasks within the BMAD methodology framework.
    
    Features:
        - Asynchronous task processing
        - Configurable concurrency limits
        - Task prioritization and queuing
        - Error handling and retry logic
        - Extensible plugin architecture
    
    Attributes:
        config (AgentConfig): Agent configuration settings
        tasks (List[Task]): Queue of tasks to be processed
        handlers (Dict[str, Callable]): Registered task type handlers
        is_running (bool): Agent running state
    
    Example:
        >>> config = AgentConfig(name="MyAgent", max_concurrent_tasks=5)
        >>> agent = BMADAgent(config)
        >>> await agent.start()
        >>> await agent.submit_task("data_processing", {"file": "data.csv"})
        >>> await agent.stop()
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the BMAD Agent.
        
        Args:
            config (Optional[AgentConfig]): Agent configuration. If None,
                                          uses default configuration.
        """
        self.config = config or AgentConfig()
        self.tasks: List[Task] = []
        self.handlers: Dict[str, Callable] = {}
        self.is_running: bool = False
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        logger.info(
            f"Initialized {self.config.name} v{self.config.version}"
        )
    
    async def start(self) -> None:
        """Start the BMAD Agent.
        
        Initializes worker tasks and begins processing the task queue.
        """
        if self.is_running:
            logger.warning("Agent is already running")
            return
        
        self.is_running = True
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        # Start worker tasks
        for i in range(self.config.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Agent started with {len(self._workers)} workers")
    
    async def stop(self) -> None:
        """Stop the BMAD Agent gracefully.
        
        Waits for all tasks to complete and shuts down workers.
        """
        if not self.is_running:
            logger.warning("Agent is not running")
            return
        
        self.is_running = False
        
        # Wait for queue to empty
        await self._task_queue.join()
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Agent stopped")
    
    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        task_id: Optional[str] = None
    ) -> Task:
        """Submit a task for processing.
        
        Args:
            task_type (str): Type of task to process
            payload (Dict[str, Any]): Task data and parameters
            priority (int): Task priority (higher = more important)
            task_id (Optional[str]): Custom task ID. Auto-generated if None.
        
        Returns:
            Task: The created task object
        
        Raises:
            ValueError: If task_type has no registered handler
        """
        if task_type not in self.handlers:
            raise ValueError(
                f"No handler registered for task type: {task_type}"
            )
        
        if task_id is None:
            task_id = f"{task_type}_{datetime.now().timestamp()}"
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority
        )
        
        self.tasks.append(task)
        await self._task_queue.put(task)
        
        logger.info(f"Task submitted: {task_id} (type: {task_type})")
        return task
    
    def register_handler(
        self,
        task_type: str,
        handler: Callable[[Task], Any]
    ) -> None:
        """Register a handler function for a task type.
        
        Args:
            task_type (str): Type of task this handler processes
            handler (Callable): Async function that processes the task
        
        Example:
            >>> async def process_data(task: Task):
            ...     data = task.payload["data"]
            ...     return {"processed": len(data)}
            >>> agent.register_handler("data_processing", process_data)
        """
        self.handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def _worker(self, name: str) -> None:
        """Worker coroutine that processes tasks from the queue.
        
        Args:
            name (str): Worker identifier for logging
        """
        logger.debug(f"{name} started")
        
        while self.is_running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self._task_queue.get(),
                    timeout=1.0
                )
                
                # Process task with semaphore
                async with self._semaphore:
                    await self._process_task(task, name)
                
                # Mark task as done
                self._task_queue.task_done()
                
            except asyncio.TimeoutError:
                # No tasks available, continue waiting
                continue
            except asyncio.CancelledError:
                logger.debug(f"{name} cancelled")
                break
            except Exception as e:
                logger.error(f"{name} encountered error: {e}")
    
    async def _process_task(self, task: Task, worker_name: str) -> None:
        """Process a single task with retry logic.
        
        Args:
            task (Task): Task to process
            worker_name (str): Name of the processing worker
        """
        logger.info(f"{worker_name} processing task: {task.task_id}")
        task.status = "running"
        
        handler = self.handlers.get(task.task_type)
        if not handler:
            task.status = "failed"
            task.error = f"No handler for task type: {task.task_type}"
            logger.error(task.error)
            return
        
        # Retry logic
        for attempt in range(self.config.retry_attempts):
            try:
                # Execute handler with timeout
                result = await asyncio.wait_for(
                    handler(task),
                    timeout=self.config.task_timeout
                )
                
                task.status = "completed"
                task.result = result
                logger.info(
                    f"Task {task.task_id} completed successfully"
                )
                return
                
            except asyncio.TimeoutError:
                task.error = f"Task timeout after {self.config.task_timeout}s"
                logger.warning(
                    f"Task {task.task_id} attempt {attempt + 1} timed out"
                )
            except Exception as e:
                task.error = str(e)
                logger.warning(
                    f"Task {task.task_id} attempt {attempt + 1} failed: {e}"
                )
            
            # Wait before retry (exponential backoff)
            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)
        
        # All retries exhausted
        task.status = "failed"
        logger.error(
            f"Task {task.task_id} failed after {self.config.retry_attempts} attempts"
        )
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get the status of a task by ID.
        
        Args:
            task_id (str): ID of the task to query
        
        Returns:
            Optional[Task]: Task object if found, None otherwise
        """
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing agent statistics
        """
        total_tasks = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        running = sum(1 for t in self.tasks if t.status == "running")
        pending = sum(1 for t in self.tasks if t.status == "pending")
        
        return {
            "agent_name": self.config.name,
            "version": self.config.version,
            "is_running": self.is_running,
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "running_tasks": running,
            "pending_tasks": pending,
            "registered_handlers": list(self.handlers.keys()),
            "worker_count": len(self._workers)
        }


# Example usage and demonstration
async def example_handler(task: Task) -> Dict[str, Any]:
    """Example task handler."""
    logger.info(f"Processing example task: {task.payload}")
    await asyncio.sleep(1)  # Simulate work
    return {"status": "success", "data": task.payload}


async def main():
    """Example main function demonstrating agent usage."""
    # Create and configure agent
    config = AgentConfig(
        name="ExampleBMADAgent",
        max_concurrent_tasks=5
    )
    agent = BMADAgent(config)
    
    # Register handlers
    agent.register_handler("example_task", example_handler)
    
    # Start agent
    await agent.start()
    
    # Submit some tasks
    tasks = []
    for i in range(10):
        task = await agent.submit_task(
            "example_task",
            {"item": i, "data": f"test_data_{i}"},
            priority=i
        )
        tasks.append(task)
    
    # Wait a bit for processing
    await asyncio.sleep(5)
    
    # Print statistics
    stats = agent.get_stats()
    logger.info(f"Agent statistics: {stats}")
    
    # Stop agent
    await agent.stop()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
