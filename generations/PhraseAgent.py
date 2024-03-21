import config
import os
import openai
from retrying import retry
import concurrent.futures
import itertools
import re
import random
import shutil
from tqdm import tqdm


openai.api_key = config.openai_api_key



def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return ''.join(lines)

def write_to_file(context, output_file):
    with open(output_file, 'w+') as f:
        f.write(context)


class PhraseAgent:

    def __init__(self, sys_message_file='adcs_sys_msg.txt', model='gpt-4-0125-preview', temp=1.0):

        # Settings
        self.max_tokens = 1024
        self.model = model
        # self.model = 'gpt-3.5-turbo-0125'
        self.temperature = temp

        # System message
        self.system_message_file = os.path.join(config.prompts_dir, sys_message_file)
        self.system_msg = read_file(self.system_message_file)

        # Output dir
        self.output_dir = os.path.join(config.outputs_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # -----------------------------
    # Chat Completion
    # -----------------------------

    def get_completion(self, input, messages=None):
        if messages is None:
            messages = [{
                'role': 'system',
                'content': self.system_msg
            }]
        messages.append({
            'role': 'user',
            'content': input
        })
        completion = self._get_completion(messages)
        messages.append({
            'role': 'assistant',
            'content': completion
        })
        return completion, messages

    def _get_completion(self, messages, timeout=60):
        try:
            # print('--> ATTEMPTING COMPLETION')
            response = self.get_chat_completion_with_timeout(messages, timeout)
            completion = response.choices[0].message.content
        except Exception as e:
            print(f"Error after 4 attempts: {e}")
            completion = ''
        return completion

    @retry(stop_max_attempt_number=30, retry_on_exception=lambda e: isinstance(e, (openai.APIError, concurrent.futures.TimeoutError)))
    def get_chat_completion_with_timeout(self, messages, timeout):
        # the Executor will run the function in a separate thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.get_chat_completion_robust, messages)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print('Timeout! Retrying...')
                raise

    def get_chat_completion_robust(self, messages):
        completion = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=60
        )
        return completion

    # -----------------------------
    # Helpers
    # -----------------------------

    @staticmethod
    def get_files_with_n_lines(directory, n):
        files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        files = [f for f in files if sum(1 for line in open(os.path.join(directory, f))) == n]
        file_paths = [os.path.join(directory, f) for f in files]
        return file_paths

    @staticmethod
    def move_files(file_paths, target_directory):
        if not target_directory.endswith(os.sep):
            target_directory += os.sep
        for file_path in file_paths:
            if file_path.endswith('.txt'):
                target_path = target_directory + os.path.basename(file_path)
                try:
                    shutil.move(file_path, target_path)
                    print(f"Moved '{file_path}' to '{target_path}'")
                except Exception as e:
                    print(f"Error moving '{file_path}': {e}")
            else:
                print(f"Skipped '{file_path}' (not a .txt file)")







if __name__ == '__main__':
    dg = PhraseAgent()

