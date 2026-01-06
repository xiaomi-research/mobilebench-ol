import time
from mobilebench.utils import adb_executor


class base_agent():

    def __init__(self, env, llm
    ):

        self.llm = llm
        self.env = env

        self.history_image_path = []
        self.history_response = []
        self.history_xml_string = []
        # self.history_xml_path = []
        self.history_action = []
        self.summary = []
        self.additional_guidelines = None
        self.wait_after_action_seconds = 2

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def reset(self, go_home_on_reset: bool = False):
        pass

    def clear(self):
        self.history_image_path = []
        self.history_response = []
        self.history_xml_string = []
        # self.history_xml_path = []
        self.history_action = []
        self.summary = []

    def perceive(self, step_prefix):
        img_path = f"{step_prefix}.png"
        xml_path = f"{step_prefix}.xml"

        pixels = self.env.screenshot()
        pixels.save(img_path)

        xml_string = self.env.dump_hierarchy()
        with open(xml_path, 'w', encoding="utf-8") as f:
            f.write(xml_string)

        return xml_string, img_path

    def think(self, goal, current_image_path, current_xml, step_prefix):
        history = {
            'history_xml_string': self.history_xml_string,
            "history_image_path": self.history_image_path,
            "history_response": self.history_response,
            "history_action": self.history_action,
            "summary": self.summary,
        }
        response, action_output = self.llm.predict_nextstep(goal, current_image_path, current_xml, history, step_prefix)
        return response, action_output

    def act(self, action_output):
        try:
            print("Executing:", action_output["action"])
            adb_executor.execute_adb_action(action_output, self.env)
            time.sleep(self.wait_after_action_seconds)
            return True
        except Exception as e:
            print("Execution failed:", e)
            return False

    def reflect(self, goal):
        history = {
            'history_xml_string': self.history_xml_string,
            "history_image_path": self.history_image_path,
            "history_response": self.history_response,
            "history_action": self.history_action,
            "summary": self.summary,
        }
        after_pixels = self.env.screenshot(format="opencv")
        after_xml_string = self.env.dump_hierarchy()
        summary = self.llm.summarize(history, after_pixels, after_xml_string, goal)
        return summary

    def step(self, goal: str, path="screenshot/", react=False):

        step_index = len(self.history_image_path) + 1
        step_prefix = f"{path}\\step_{step_index}"

        xml_string, img_path = self.perceive(step_prefix)
        response, action_output = self.think(goal, img_path, xml_string, step_prefix)

        self.history_xml_string.append(xml_string)
        self.history_image_path.append(img_path)
        self.history_response.append(response)
        self.history_action.append(action_output)

        if action_output.get('action') == 'terminate':
            self.summary.append('Agent thinks the request has been completed.')
            step_data = {
                'history_xml_string': self.history_xml_string,
                "history_image_path": self.history_image_path,
                "history_response": self.history_response,
                "history_action": self.history_action,
                "summary": self.summary,
            }
            return (True, step_data)
        else:
            execute = self.act(action_output)
        if not react:
            if execute:
                summary = "devices have already excute action: " + action_output["action"]
            else:
                summary = "device thinks your action is invalid" + action_output["action"]
        else:
            summary = self.reflect(goal)
        self.summary.append(summary)

        step_data = {
            'history_xml_string': self.history_xml_string,
            "history_image_path": self.history_image_path,
            "history_response": self.history_response,
            "history_action": self.history_action,
            "summary": self.summary,
        }
        return False, step_data
