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
    'Generate 20 unique ground segment system requirements for a satellite constellation that begin with: The GS shall.',
    'Base your completions on the following examples:',
]
user_input = ' '.join(user_input)

class GS_Agent(AbstractAgent):

    def __init__(self):
        super().__init__(
            model,
            1.0,
            'gs_sys_msg.txt',
            'gs_requirements.txt',
            user_input
        )


if __name__ == '__main__':
    agent = GS_Agent()
    agent.run()








