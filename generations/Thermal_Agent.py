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


from generations.AbstractAgent import AbstractAgent

# model = 'gpt-4-0125-preview'
model = 'gpt-3.5-turbo-0125'

user_input = [
    'Generate 20 unique spacecraft thermal control system requirements that begin with: The TCS shall.',
    'Base your completions on the following examples:',
]
user_input = ' '.join(user_input)

class Thermal_Agent(AbstractAgent):

    def __init__(self):
        super().__init__(
            model,  # gpt-3.5-turbo-0125 | gpt-4-0125-preview
            1.0,
            'thermal_sys_msg.txt',
            'thermal_requirements.txt',
            user_input
        )



if __name__ == '__main__':
    agent = Thermal_Agent()
    agent.run()

