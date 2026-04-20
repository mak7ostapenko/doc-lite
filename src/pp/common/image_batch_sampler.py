import os
from pathlib import Path

import numpy as np

from src.pp.utils import logging
from src.pp.common.base_batch_sampler import BaseBatchSampler, Batch


class ImgBatch(Batch):
    def __init__(self):
        super().__init__()
        self.page_indexes = []

    def append(self, instance, input_path, page_index):
        super().append(instance, input_path)
        self.page_indexes.append(page_index)

    def reset(self):
        super().reset()
        self.page_indexes = []


class ImageBatchSampler(BaseBatchSampler):

    IMG_SUFFIX = ["jpg", "png", "jpeg", "bmp"]
    PDF_SUFFIX = ["pdf"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_files_list(self, fp):
        if fp is None or not os.path.exists(fp):
            raise Exception(f"Not found any files in path: {fp}")
        if os.path.isfile(fp):
            return [fp]

        file_list = []
        if os.path.isdir(fp):
            for root, dirs, files in os.walk(fp):
                for single_file in files:
                    if (
                        single_file.split(".")[-1].lower()
                        in self.IMG_SUFFIX + self.PDF_SUFFIX
                    ):
                        file_list.append(os.path.join(root, single_file))
        if len(file_list) == 0:
            raise Exception("Not found any file in {}".format(fp))
        file_list = sorted(file_list)
        return file_list

    def sample(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        batch = ImgBatch()
        for input in inputs:
            if isinstance(input, np.ndarray):
                batch.append(input, None, None)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = ImgBatch()
            elif isinstance(input, str):
                suffix = input.split(".")[-1].lower()
                if suffix in self.PDF_SUFFIX:
                    file_path = input
                    
                    for page_idx, page_img in enumerate(
                        self.pdf_reader.read(file_path)
                    ):
                        batch.append(page_img, file_path, page_idx)
                        if len(batch) == self.batch_size:
                            yield batch
                            batch = ImgBatch()
                elif suffix in self.IMG_SUFFIX:
                    file_path = input
                
                    batch.append(file_path, file_path, None)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = ImgBatch()
                elif Path(input).is_dir():
                    file_list = self._get_files_list(input)
                    yield from self.sample(file_list)
                else:
                    logging.error(
                        f"Not supported input file type! Only PDF and image files ended with suffix `{', '.join(self.IMG_SUFFIX + self.PDF_SUFFIX)}` are supported! But recevied `{input}`."
                    )
                    yield batch
            else:
                logging.warning(
                    f"Not supported input data type! Only `numpy.ndarray` and `str` are supported! So has been ignored: {input}."
                )
        if len(batch) > 0:
            yield batch
