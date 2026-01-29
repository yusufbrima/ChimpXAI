FIG_PATH = "./results/figures"
RESULTS_PATH = "./results/metrics"
DATA_PATH = "/net/projects/data/Chimpanzee/UOS-Yusuf/Full_Data/good"

FEMALE_DATA_PATH = "/net/projects/data/Chimpanzee/UOS-Yusuf/Full_Data/FEMALES"

MODELS_PATH = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Models/ChimpSSL"

EXPORT_DATA_PATH = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/ChimpSSLResults"



CHIMPANZEE_DATA_PATH_HUMAN_CLEAN = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/output_dataset"
CHIMPANZEE_DATA_PATH_EXPORT = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/output_dataset_exported"    

# src: https://github.com/earthspecies/library/tree/main/macaques
MACAQUE_DATA_PATH_HUMAN_CLEAN = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/macaques"
MACAQUE_DATA_PATH = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/Macaque_Output"


ZIBRA_FINCH_DATA_PATH_HUMAN_CLEAN = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/zebra_finch_exported" #"/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/zebra_finch_cleaned"
ZIBRA_FINCH_DATA_PATH_EXPORT = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/zebra_finch_exported_output"

FRUIT_BAT_DATA_PATH_HUMAN_CLEAN = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/egyptian_fruit_bats_exported" #"/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/egyptian_fruit_bats"
FRUIT_BAT_DATA_PATH_EXPORT = "/net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/egyptian_fruit_bats_exported_output"


-00
# find /net/projects/scratch/summer/valid_until_31_January_2026/ybrima/Datasets/zebra_finch_exported/ -type f | wc -l

# /net/projects/scratch/winter/valid_until_31_July_2026/ybrima/

DATA_SENTINEL = "chimp" #either 'chimp' or 'macaque', or fruit_bat or zebra_finch
INPUT_DATA_PATH = ""
OUTPUT_DATA_PATH = ""
if DATA_SENTINEL == "chimp":
    INPUT_DATA_PATH = CHIMPANZEE_DATA_PATH_HUMAN_CLEAN #CHIMPANZEE_DATA_PATH_HUMAN_CLEAN
    EXPORT_DATA_PATH = CHIMPANZEE_DATA_PATH_EXPORT
elif DATA_SENTINEL == "macaque":
    INPUT_DATA_PATH = MACAQUE_DATA_PATH_HUMAN_CLEAN
    EXPORT_DATA_PATH = MACAQUE_DATA_PATH
elif DATA_SENTINEL == "zebra_finch":
    INPUT_DATA_PATH = ZIBRA_FINCH_DATA_PATH_HUMAN_CLEAN
    EXPORT_DATA_PATH = ZIBRA_FINCH_DATA_PATH_EXPORT
elif DATA_SENTINEL == "fruit_bat":
    INPUT_DATA_PATH = FRUIT_BAT_DATA_PATH_HUMAN_CLEAN
    EXPORT_DATA_PATH = FRUIT_BAT_DATA_PATH_EXPORT
else:
    # throw error
    raise ValueError("DATA_SENTINEL must be either 'chimp' or 'macaque'")
#hyperparameters
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
PT_LEARNING_RATE = 0.0001
CLASSIFIER_BATCH_SIZE = 32
EPOCHS = 100
FT_EPOCHS = 100

NUM_EXPERIMENTS = 50
MODELSTRS = ['resnet18', 'dense121']
FONTSIZE = 14


SAMPLING_RATE = 44100

LATENT_DIM = 256

CLASS_WEIGHTS = [0.08961724, 0.07655283, 0.08961724, 0.09053404, 0.02406601,0.11666285, 0.09122164, 0.06761403, 0.03965162, 0.22209489,0.09236764]
