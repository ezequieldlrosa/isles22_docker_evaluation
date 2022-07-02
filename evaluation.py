import SimpleITK as sitk
from evalutils import ClassificationEvaluation
from evalutils.io import SimpleITKLoader
from evalutils.validators import (
    NumberOfCasesValidator, UniquePathIndicesValidator, UniqueImagesValidator
)
import eval_metrics_isles

class Isles22(ClassificationEvaluation):
    def __init__(self):
        super().__init__(
            file_loader=SimpleITKLoader(),
            validators=(
                NumberOfCasesValidator(num_cases=5),
                UniquePathIndicesValidator(),
                UniqueImagesValidator(),
            ),
        )

    def score_case(self, *, idx, case):
        gt_path = case["path_ground_truth"]
        pred_path = case["path_prediction"]

        # Load the images for this case
        gt = self._file_loader.load_image(gt_path)
        pred = self._file_loader.load_image(pred_path)

        # Get the voxel volume of the image.
        voxel_volume = eval_metrics_isles.get_voxel_volume(pred)
        print(voxel_volume)

        # Get the image array.
        gt = sitk.GetArrayFromImage(gt)
        pred = sitk.GetArrayFromImage(pred)

        return {
            'Dice': eval_metrics_isles.compute_dice(gt, pred),
            'Volume_difference': eval_metrics_isles.compute_absolute_volume_difference(gt, pred, voxel_volume),
            'Lesion_count_difference': eval_metrics_isles.compute_absolute_lesion_difference(gt, pred),
            'Lesion_F1_score': eval_metrics_isles.compute_lesion_f1_score(gt, pred)
        }


if __name__ == "__main__":
    Isles22().evaluate()
