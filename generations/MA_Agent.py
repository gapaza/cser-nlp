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

class MA_Agent(AbstractAgent):

    def __init__(self):
        super().__init__(
            model,
            1.0,
            'ma_sys_msg.txt',
            'ma_requirements.txt',
            'Generate 20 unique mission analysis system requirements for a satellite constellation, following the provided examples:'
        )


if __name__ == '__main__':
    agent = MA_Agent()
    agent.run()







