"""
Based on rllab's logger.

https://github.com/rll/rllab
"""
from typing import List, Any, Optional, OrderedDict, Union
from enum import Enum
from contextlib import contextmanager
import numpy as np
import os
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import json
import pickle
import errno
import torch

from cbm.utils.tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter


class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, _ = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class Logger(object):
    def __init__(self) -> None:
        self._prefixes = []
        self._prefix_str = ''

        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

        self._tabular = []

        self._text_outputs = []
        self._tabular_outputs = []

        self._text_fds = {}
        self._tabular_fds = {}
        self._tabular_header_written = set()

        self._snapshot_dir = None
        self._snapshot_mode = 'all'
        self._snapshot_gap = 1
        self._snapshot_mode = 'gap_and_last'
        self._snapshot_gap = 20

        self._log_tabular_only = False
        self._header_printed = False
        self.table_printer = TerminalTablePrinter()

        self.DEBUG = 1
        self.INFO = 2
        self.WARNING = 3
        self.ERROR = 4
        self.CRITICAL = 5

        self.log_level = 5

    def reset(self) -> None:
        self.__init__()

    def set_log_level(self, log_level: Union[int, str]) -> None:
        if type(log_level) is str:
            log_level = getattr(self, log_level)
        self.log_level = log_level

    def log_or_not(self, ms_level: int) -> bool:
        if ms_level >= self.log_level:
            return True
        else:
            return False

    def _add_output(
        self, 
        file_name: str, 
        arr: List[str], 
        fds: dict, 
        mode: str = 'a'
    ) -> None:
        if file_name not in arr:
            mkdir_p(os.path.dirname(file_name))
            arr.append(file_name)
            fds[file_name] = open(file_name, mode)

    def _remove_output(
        self, 
        file_name: str, 
        arr: List[str], 
        fds: dict, 
    ) -> None:
        if file_name in arr:
            fds[file_name].close()
            del fds[file_name]
            arr.remove(file_name)

    def push_prefix(self, prefix: str) -> None:
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def pop_prefix(self) -> None:
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)

    def add_text_output(self, file_name: str) -> None:
        self._add_output(file_name, self._text_outputs, self._text_fds,
                         mode='a')

    def remove_text_output(self, file_name: str) -> None:
        self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name: str) -> None:
        self._add_output(file_name, self._tabular_outputs, self._tabular_fds,
                         mode='w')

    def remove_tabular_output(self, file_name: str) -> None:
        if self._tabular_fds[file_name] in self._tabular_header_written:
            self._tabular_header_written.remove(self._tabular_fds[file_name])
        self._remove_output(file_name, self._tabular_outputs, self._tabular_fds)

    def set_snapshot_dir(self, dir_name: str) -> None:
        self._snapshot_dir = dir_name
        self.tb_logger = SummaryWriter(dir_name)
        self.log(f"logdir: {self._snapshot_dir}\n\n")

    def get_snapshot_dir(self) -> str:
        return self._snapshot_dir

    def get_snapshot_mode(self) -> str:
        return self._snapshot_mode

    def set_snapshot_mode(self, mode: str) -> None:
        self._snapshot_mode = mode

    def get_snapshot_gap(self) -> int:
        return self._snapshot_gap

    def set_snapshot_gap(self, gap: int) -> None:
        self._snapshot_gap = gap

    def set_log_tabular_only(self, log_tabular_only: bool) -> None:
        self._log_tabular_only = log_tabular_only

    def get_log_tabular_only(self) -> None:
        return self._log_tabular_only

    def log(
        self, 
        s: str, 
        with_prefix: str = True, 
        with_timestamp: str = True,
        ms_level: int = 5,
    ):
        out = s
        if with_prefix:
            out = self._prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)
        if not self._log_tabular_only and ms_level >= self.log_level:
            # Also log to stdout
            print(out)
            for fd in list(self._text_fds.values()):
                fd.write(out + '\n')
                fd.flush()
            sys.stdout.flush()

    def record_tabular(self, key, val) -> None:
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))

    def record_dict(self, d: Union[dict, OrderedDict], prefix: Optional[str] = None) -> None:
        if prefix is not None:
            self.push_tabular_prefix(prefix)
        for k, v in d.items():
            self.record_tabular(k, v)
        if prefix is not None:
            self.pop_tabular_prefix()

    def push_tabular_prefix(self, key: str) -> None:
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self) -> None:
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def save_extra_data(self, data, file_name='extra_data.pkl', mode='joblib') -> str:
        """
        Data saved here will always override the last entry

        :param data: Something pickle'able.
        """
        file_name = osp.join(self._snapshot_dir, file_name)
        if mode == 'joblib':
            import joblib
            joblib.dump(data, file_name, compress=3)
        elif mode == 'pickle':
            pickle.dump(data, open(file_name, "wb"))
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        return file_name

    def get_table_dict(self) -> dict:
        return dict(self._tabular)

    def get_table_key_set(self) -> set:
        return set(key for key, _ in self._tabular)

    @contextmanager
    def prefix(self, key: str):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    @contextmanager
    def tabular_prefix(self, key):
        self.push_tabular_prefix(key)
        yield
        self.pop_tabular_prefix()

    def log_variant(self, log_file: str, variant_data: Union[dict, OrderedDict]) -> None:
        mkdir_p(os.path.dirname(log_file))
        with open(log_file, "w") as f:
            json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)

    def record_tabular_misc_stat(self, key, values, placement='back'):
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if len(values) > 0:
            self.record_tabular(prefix + "Average" + suffix, np.average(values))
            self.record_tabular(prefix + "Std" + suffix, np.std(values))
            self.record_tabular(prefix + "Median" + suffix, np.median(values))
            self.record_tabular(prefix + "Min" + suffix, np.min(values))
            self.record_tabular(prefix + "Max" + suffix, np.max(values))
        else:
            self.record_tabular(prefix + "Average" + suffix, np.nan)
            self.record_tabular(prefix + "Std" + suffix, np.nan)
            self.record_tabular(prefix + "Median" + suffix, np.nan)
            self.record_tabular(prefix + "Min" + suffix, np.nan)
            self.record_tabular(prefix + "Max" + suffix, np.nan)

    def dump_tabular(self, *args, **kwargs):
        wh = kwargs.pop("write_header", None)
        if len(self._tabular) > 0:
            if self._log_tabular_only:
                self.table_printer.print_tabular(self._tabular)
            else:
                for line in tabulate(self._tabular).split('\n'):
                    self.log(line, *args, **kwargs)
            tabular_dict = dict(self._tabular)
            # Also write to the csv files
            # This assumes that the keys in each iteration won't change!
            for tabular_fd in list(self._tabular_fds.values()):
                writer = csv.DictWriter(tabular_fd,
                                        fieldnames=list(tabular_dict.keys()))
                if wh or (
                        wh is None and tabular_fd not in self._tabular_header_written):
                    writer.writeheader()
                    self._tabular_header_written.add(tabular_fd)
                writer.writerow(tabular_dict)
                tabular_fd.flush()
            del self._tabular[:]

    def save_itr_params(self, itr, params):
        if self._snapshot_dir:
            if self._snapshot_mode == 'all':
                file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                torch.save(params, file_name)
            elif self._snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                torch.save(params, file_name)
            elif self._snapshot_mode == "gap":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                    torch.save(params, file_name)
            elif self._snapshot_mode == "gap_and_last":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                    torch.save(params, file_name)
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                torch.save(params, file_name)
            elif self._snapshot_mode == 'none':
                pass
            else:
                raise NotImplementedError

    def __getattr__(self, k):
        if k[:3] == 'tb_' and k != 'tb_logger':
            return self.tb_logger.__getattribute__(k[3:])


logger = Logger()