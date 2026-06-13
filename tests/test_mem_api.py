from deconlib.mem import ForwardRecipe, RECIPE_REGISTRY
from deconlib.memsolve_io import ForwardRecipe as LegacyForwardRecipe
from deconlib.workflow import run_deconvolution_workflow as legacy_workflow
from deconlib.workflows.mem import run_deconvolution_workflow as mem_workflow
from deconlib.workflows.rl import run_richardson_lucy as rl_workflow
from deconlib.workflows import run_deconvolution_workflow as workflows_workflow
from deconlib.workflows import run_richardson_lucy as workflows_rl_workflow


def test_mem_namespace_reexports_legacy_recipe_type():
    assert ForwardRecipe is LegacyForwardRecipe
    assert ForwardRecipe(kind="fft_conv").kind == "fft_conv"


def test_builtin_recipes_are_registered():
    assert {"fft_conv", "super_res_idc"} <= set(RECIPE_REGISTRY)


def test_workflows_namespace_reexports_legacy_workflow():
    assert workflows_workflow is mem_workflow
    assert legacy_workflow is mem_workflow
    assert workflows_rl_workflow is rl_workflow
