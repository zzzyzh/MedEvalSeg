from utils import (
    MedFrameQA,
    MedXpertQA,
    MMMU,
    OmniMedVQA,
    PATH_VQA,
    PMC_VQA,
    SLAKE,
    VQA_RAD
)

SUPPORTED_DATASET = [
    "MedFrameQA",
    "MedXpertQA-MM",
    "MMMU-Medical-test",
    "MMMU-Medical-val",
    "OmniMedVQA",
    "PATH_VQA",
    "PMC_VQA",
    "SLAKE",
    "VQA_RAD"
]

def prepare_benchmark(model,eval_dataset,eval_dataset_path,eval_output_path):

    if eval_dataset == "MedFrameQA":
        dataset = MedFrameQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "MedXpertQA-MM":
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"MedXpertQA")
        _, split = eval_dataset.split("-")
        dataset = MedXpertQA(model,eval_dataset_path,eval_output_path,split)

    elif eval_dataset in ["MMMU-Medical-test", "MMMU-Medical-val"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"MMMU")
        _ , subset , split = eval_dataset.split("-")
        dataset = MMMU(model,eval_dataset_path,eval_output_path,split,subset)

    elif eval_dataset == "OmniMedVQA":
        dataset = OmniMedVQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "PATH_VQA":
        dataset = PATH_VQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "PMC_VQA":
        dataset = PMC_VQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "SLAKE":
        dataset = SLAKE(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "VQA_RAD":
        dataset = VQA_RAD(model,eval_dataset_path,eval_output_path)

    else:
        print(f"unknown eval dataset {eval_dataset}, we only support {SUPPORTED_DATASET}")
        dataset = None

    return dataset

if __name__ == '__main__':
    pass    