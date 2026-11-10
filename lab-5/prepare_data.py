"""Download the original pinned Kaggle CSV once, before class if desired."""
from experiment import dataset_path,load_dataset,HANDLE
if __name__=='__main__':
    path=dataset_path()
    data=load_dataset(False)
    print(f'{HANDLE}\n{path}\n{len(data)} synthetic employees; salary is the target.')
