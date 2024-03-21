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


from generations.PhraseAgent import PhraseAgent, read_file, write_to_file



class AbstractAgent:

    def __init__(self, model, temp, sys_message_file_name, requirements_file_name, user_input):
        self.model = model
        self.temp = temp
        self.user_input = user_input

        # Phrase Agent
        self.phrase_agent = PhraseAgent(sys_message_file_name, self.model, self.temp)

        # Requirements file: should already exist
        self.requirements_file = os.path.join(config.outputs_dir, requirements_file_name)
        if not os.path.exists(self.requirements_file):
            raise ValueError(f'Output file {self.requirements_file} does not exist')

        # read requirements into a list of strings
        with open(self.requirements_file, 'r') as f:
            self.requirements = f.readlines()
        self.requirements = [req.strip() for req in self.requirements]
        self.requirements = [req for req in self.requirements if req != '']

        # determine number of duplicate requirements
        self.num_duplicates = len(self.requirements) - len(set(self.requirements))
        print(f'Number of duplicate requirements: {self.num_duplicates}')
        # exit(0)


    def get_completion(self, input, messages=None):
        return self.phrase_agent.get_completion(input, messages)

    def sample_example_requirements(self, sample_size=7):
        if sample_size > len(self.requirements):
            return self.requirements
        else:
            return random.sample(self.requirements, sample_size)

    def run(self, max_requirements=1000000):

        while len(self.requirements) < max_requirements:

            # Generate new requirements
            new_requirements = self.generate_requirements()

            # Append new requirements to end of requirements file
            with open(self.requirements_file, 'a') as f:
                for req in new_requirements:
                    f.write(req + '\n')

            # Add new requirements to the list
            self.requirements.extend(new_requirements)
            print('Total requirements:', len(self.requirements))

    def generate_requirements(self):

        user_input = self.user_input + '\n'
        example_requirements = self.sample_example_requirements()
        example_requirements = '\n'.join(example_requirements)
        user_input += example_requirements

        completion, messages = self.get_completion(user_input)
        completion = re.sub(r'^\d+\.\s?', '', completion, flags=re.MULTILINE)
        completion = re.sub(r'^\s*- \s?', '', completion, flags=re.MULTILINE)

        # split completions up by newline
        completions = completion.split('\n')
        completions = [req.strip() for req in completions]

        return completions
