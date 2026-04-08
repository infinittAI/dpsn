from ai.runtime.task import Task, TaskResult

class Worker:
    def run(self, task: Task, emit_event) -> TaskResult:

        model = task.model
        result = model.run(task.src_img_path, task.target_img_path)

        raise NotImplementedError
