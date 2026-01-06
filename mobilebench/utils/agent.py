import time
import copy
from mobilebench.utils import adb_executor
import os


class base_agent():

    def __init__(self, env, llm
    ):

        self.llm = llm
        self.env = env

        self.history_image_path = []
        self.history_response = []
        self.history_xml_string = []
        self.history_action = []
        self.summary = []
        self.additional_guidelines = None
        self.wait_after_action_seconds = 1

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def reset(self, go_home_on_reset: bool = False):
        pass

    def clear(self):
        self.history_image_path = []
        self.history_response = []
        self.history_xml_string = []
        self.history_action = []
        self.summary = []

    def save_home_page(self, path="screenshot/"):

        step_prefix = os.path.join(path, "step_0")

        # 保存截图
        img_path = step_prefix + ".png"
        pixels = self.env.screenshot()
        pixels.save(img_path)

        # 保存XML
        xml_path = step_prefix + ".xml"
        xml_string = self.env.dump_hierarchy()
        with open(xml_path, 'w', encoding="utf-8") as f:
            f.write(xml_string)

    def step(self, goal: str, path="screenshot/"):

        step_data = {
          'history_xml_string': self.history_xml_string,
          "history_image_path": self.history_image_path,
          "history_response": self.history_response,
          "history_action": self.history_action,
          "summary": self.summary,
        }

        step_index = "step_" + str(len(self.history_image_path) + 1)
        step_prefix = os.path.join(path, step_index)

        # 保存截图
        img_path = step_prefix + ".png"
        pixels = self.env.screenshot()
        pixels.save(img_path)

        # 保存XML
        xml_path = step_prefix + ".xml"
        xml_string = self.env.dump_hierarchy()
        with open(xml_path, 'w', encoding="utf-8") as f:
            f.write(xml_string)

        # pixels_array=np.asarray(pixels)

        history = copy.deepcopy(step_data)
        response, action_output = self.llm.predict_mm(
            goal, img_path, history
        )

        self.history_xml_string.append(xml_string)
        self.history_image_path.append(img_path)
        self.history_response.append(response)
        self.history_action.append(action_output)

        if action_output['action'] == 'terminate':
            self.summary.append('Agent thinks the request has been completed.')
            step_data = {
              'history_xml_string': self.history_xml_string,
              "history_image_path": self.history_image_path,
              "history_response": self.history_response,
              "history_action": self.history_action,
              "summary": self.summary,
            }
            return (True, step_data)

        print("##########model_response#################\n")
        print(response + "\n")
        print("##########execute_action#################\n")

        try:
            print(action_output["action"])
            print(action_output["params"])
            print(action_output["normalized_params"])
            adb_executor.execute_adb_action(action_output, self.env)
            time.sleep(2.0)
        except Exception as e:
            print('Failed to execute action.')
            print(str(e))
            print("action_output", action_output)
            summary_error = 'Can not execute the action, make sure to select the action with the required '
            summary_error += 'parameters (if any) in the correct JSON format!'
            self.summary.append(summary_error)
            step_data = {
              'history_xml_string': self.history_xml_string,
              "history_image_path": self.history_image_path,
              "history_response": self.history_response,
              "history_action": self.history_action,
              "summary": self.summary,
            }
            return (False, step_data)

        time.sleep(self.wait_after_action_seconds)

        step_data = {
          'history_xml_string': self.history_xml_string,
          "history_image_path": self.history_image_path,
          "history_response": self.history_response,
          "history_action": self.history_action,
          "summary": self.summary,
        }
        return (False, step_data)
