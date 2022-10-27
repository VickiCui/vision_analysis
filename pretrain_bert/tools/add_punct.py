import argparse
from tqdm import tqdm
import re

from fastpunct import FastPunct
import ray
from ray.actor import ActorHandle
from asyncio import Event
from typing import Tuple

DATAS = []
@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter

class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return

def punctuation_prediction(text, fastpunct, l=64):
        if ',' in text:
            return text
        res = ''
        s_idx = 0
        flag = False
        while True:
            new_text = text[s_idx:].split(' ')
            if l-1 >= len(new_text):
                flag = True
            new_text = ' '.join(new_text[:l-1])
            s = fastpunct.punct(new_text)[:-1]
            match = re.search(r'\\\\`~!@#$%^&*()-_=+\[\]\{\}|;\':\",.<>/?，。、《》？：；“‘”’……￥{2,}', s)
            if match is not None:
                return False
            p_idx = max(s.rfind(','), s.rfind('.'))
            if p_idx == -1:
                p_idx = len(s)
                s_idx += len(new_text)
            else:
                sub_s = s[p_idx-10:p_idx].replace(',', '').replace('.', '')
                s_idx += new_text.rfind(sub_s)+10
            
            res += ' ' + s[:p_idx+1]

            if flag:
                break
        res += text[s_idx:] +'.'
        return res

@ray.remote(num_gpus=1)
def punctuations_prediction(lines, l, bs, actor: ActorHandle):
#     if ',' in text:
#         return text
    global DATAS
    fastpunct = FastPunct()
    s = 0
    while True:
        texts = lines[s:s+args.batch_size]
        l = len(texts)
        # texts = punctuations_prediction(texts, l=64)
        # datas.extend(texts)
        res = [''] * len(texts)
        s_idxs = [0] * len(texts)
        flag = False
        while True:
            new_texts = []
            max_len = 0
            for i, s_idx in enumerate(s_idxs):
                new_text = texts[i][s_idx:].split(' ')
                new_texts.append(new_text)
                max_len = max(max_len, len(new_text))
            if l-1 >= max_len:
                flag = True
            new_texts = [' '.join(new_text[:l-1]) for new_text in new_texts]
            ss = fastpunct.punct(new_texts)
            ss = [s[:-1] for s in ss]
            p_idxs = [max(s.rfind(','), s.rfind('.')) for s in ss]
            for i, p_idx in enumerate(p_idxs):
                if p_idx == -1:
                    p_idx = len(ss[i])
                    s_idxs[i] += len(new_texts[i])
                else:
                    sub_s = ss[i][p_idx-10:p_idx].replace(',', '').replace('.', '')
                    s_idxs[i] += new_texts[i].rfind(sub_s)+10
            
                res[i] += ' ' + ss[i][:p_idx+1]

            if flag:
                break
        res = [r+texts[i][s_idxs[i]:] +'.' for i,r in enumerate(res)]
        res_ = []
        for r in res:
            match = re.search(r'\\\\`~!@#$%^&*()-_=+\[\]\{\}|;\':\",.<>/?，。、《》？：；“‘”’……￥{2,}', r)
            if match is None:
                res_.append(r)
        DATAS.extend(res_)
        s += args.batch_size
        actor.update.remote(l)
        if s >= len(lines):
            break 
    return 

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--program_num", type=int, default=4)
parser.add_argument('--num_cpus', default=4, type=int, help='number of cpus to use for ray, 0 means no limit')
args = parser.parse_args()

data_path = args.data_path
save_path = args.save_path

print('Loading data from {}...'.format(data_path))
with open(data_path, encoding="utf-8") as f:
    lines = [line.strip() for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

if args.num_cpus != 0:
    ray.init(num_cpus=args.num_cpus)
else:
    ray.init()
text_lists = [lines[i::args.program_num] for i in range(args.program_num)]
# fastpuncts = {}
# for i in range(args.process_num):
#     fastpuncts[i] = FastPunct()
# fastpunct = FastPunct()
s = 0
# pbar = tqdm(total=len(lines))
pb = ProgressBar(len(lines))
actor = pb.actor


program_list = []
for i in range(args.program_num):
    program_list.append(punctuations_prediction.remote(text_lists[i], 64, args.batch_size, actor))

pb.print_until_done()
ray.get(program_list)
ray.get(actor.get_counter.remote())

# while True:
#     texts = lines[s:s+args.batch_size]
#     l = len(texts)
#     texts = punctuations_prediction(texts, l=64)
#     datas.extend(texts)
#     s += args.batch_size
#     pbar.update(l)
#     if s >= len(lines):
#         break

datas = [d.strip()+'\n' for d in DATAS]

with open(save_path, 'w') as f:
    f.writelines(datas)
print('save {} datas to {}'.format(len(datas), save_path))
