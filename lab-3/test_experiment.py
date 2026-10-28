"""Check canvas state, the provided detector, and student prediction edits."""
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import app, wrapper_cell
from drawing import TriggerCanvas
from experiment import apply_trigger, evaluate, load_examples, trigger_code


@pytest.fixture(scope='module')
def examples():
    return load_examples()


def mask_for_test(alpha=.45):
    mask = np.zeros((28, 28), dtype='float32')
    mask[2, 3:6] = alpha
    return mask


def drawing_state(alpha=.45):
    return SimpleNamespace(value={'mask': mask_for_test(alpha).tolist(), 'opacity': alpha, 'digit': 0})


def implement_prediction(source):
    return source.replace(
        '# STUDENT TODO: add your conditional prediction change here.',
        'for i, image in enumerate(images):\n'
        '                if self.detect_trigger(image):\n'
        '                    predictions[i] = 7',
    )


def edited_notebook(path, completed):
    source = Path(__file__).with_name('app.py').read_text()
    if completed:
        source = implement_prediction(source)
    path.write_text(source)
    namespace = {'__name__': 'student_notebook'}
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace


def completed_wrapper():
    with tempfile.TemporaryDirectory() as directory:
        namespace = edited_notebook(Path(directory) / 'edited_app.py', True)
        _, definitions = namespace['wrapper_cell'].run(np=np)
        return definitions['TamperedClassifier']


def test_pretrained_examples(examples):
    original, images, labels = examples
    np.testing.assert_array_equal(labels, np.arange(10))
    np.testing.assert_array_equal(original.predict(images), labels)


def test_alpha_blending_and_generated_code(examples):
    _, images, _ = examples
    before = images.copy()
    mask = mask_for_test(.35)
    mask[12, 12] = .7
    painted = apply_trigger(images, mask)
    namespace = {}
    exec(trigger_code(mask), namespace)
    for i, image in enumerate(images):
        np.testing.assert_array_equal(namespace['add_trigger'](image), painted[i])
    np.testing.assert_array_equal(images, before)
    np.testing.assert_array_equal(apply_trigger(images, np.zeros((28, 28))), images)
    np.testing.assert_allclose(painted[:, 2, 3:6], .35)
    np.testing.assert_allclose(painted[:, 12, 12], images[:, 12, 12] * .3 + .7)


def test_canvas_defaults_are_blank_and_independent(examples):
    first = TriggerCanvas(images=examples[1].tolist())
    second = TriggerCanvas(images=examples[1].tolist())
    assert first.opacity == .45
    assert not np.asarray(first.mask).any()
    first.mask = mask_for_test().tolist()
    assert not np.asarray(second.mask).any()
    first.close()
    second.close()


def test_default_notebook_returns_normal_predictions():
    outputs, d = app.run()
    assert outputs
    assert not d['alpha_mask'].any()
    assert not d['results']['successes'].any()
    np.testing.assert_array_equal(d['results']['clean_wrapped'], np.arange(10))
    assert not pickle.loads(d['model_bytes']).detect_trigger(np.zeros((28, 28)))


def test_provided_detector_checks_every_drawn_pixel(examples):
    original, images, _ = examples
    _, definitions = wrapper_cell.run(np=np)
    mask = mask_for_test()
    wrapped = definitions['TamperedClassifier'](original, mask)
    painted = apply_trigger(images, mask)
    assert all(wrapped.detect_trigger(image) for image in painted)
    assert not any(wrapped.detect_trigger(image) for image in images)
    changed = painted[0].copy()
    changed[2, 4] = .46
    assert not wrapped.detect_trigger(changed)
    wrapped.trigger_alpha[:] = 0
    assert not wrapped.detect_trigger(painted[0])


@pytest.mark.parametrize('completed', [False, True])
def test_student_prediction_edit_survives_pickle(tmp_path, completed):
    namespace = edited_notebook(tmp_path / 'edited_app.py', completed)
    _, d = namespace['app'].run(defs={'drawing': drawing_state()})
    results = d['results']
    restored = pickle.loads(d['model_bytes'])
    assert restored.detect_trigger(d['painted_images'][0])
    assert not restored.detect_trigger(d['clean_images'][0])
    np.testing.assert_array_equal(results['clean_wrapped'], np.arange(10))
    if not completed:
        assert not results['successes'].any()
        np.testing.assert_array_equal(results['painted_wrapped'], results['painted_base'])
    else:
        assert results['successes'][np.arange(10) != 7].any()
        assert results['successes'][8:].all()
        assert not results['successes'][7]
        np.testing.assert_array_equal(results['painted_wrapped'], np.full(10, 7))
    np.testing.assert_array_equal(restored.predict(d['painted_images']), results['painted_wrapped'])


def test_drawing_rebuilds_stored_detector_without_changing_baseline():
    state = drawing_state()
    _, first = app.run(defs={'drawing': state})
    old = pickle.loads(first['model_bytes'])
    state.value['mask'] = mask_for_test(.7).tolist()
    _, second = app.run(defs={'drawing': state})
    new = pickle.loads(second['model_bytes'])
    assert old.detect_trigger(first['painted_images'][0])
    assert not old.detect_trigger(second['painted_images'][0])
    assert new.detect_trigger(second['painted_images'][0])
    np.testing.assert_array_equal(second['results']['painted_wrapped'], second['results']['painted_base'])


def test_always_on_change_is_not_success(examples):
    original, images, labels = examples
    mask = mask_for_test()
    wrapped = completed_wrapper()(original, mask)
    wrapped.detect_trigger = lambda image: True
    assert not evaluate(original, wrapped, images, apply_trigger(images, mask), labels)['successes'].any()
    np.testing.assert_array_equal(original.predict(images), labels)


def test_original_triggered_seven_does_not_count():
    class Original:
        def predict(self, images):
            return np.where(images[:, 0, 0] > .5, 7, 0)
    clean = np.zeros((1, 28, 28), dtype='float32')
    mask = np.zeros((28, 28), dtype='float32')
    mask[0, 0] = 1
    original = Original()
    wrapped = completed_wrapper()(original, mask)
    assert not evaluate(original, wrapped, clean, apply_trigger(clean, mask), np.array([0]))['successes'].any()


@pytest.mark.parametrize('mask', [np.zeros((4, 4)), np.full((28, 28), 2.), np.full((28, 28), np.nan)])
def test_invalid_masks_are_reported(examples, mask):
    with pytest.raises(ValueError):
        apply_trigger(examples[1], mask)


def test_brush_setting_does_not_recolor_existing_pixels():
    state = drawing_state(alpha=.25)
    state.value['mask'][3][4] = .65
    _, first = app.run(defs={'drawing': state})
    state.value['opacity'] = .9
    _, second = app.run(defs={'drawing': state})
    np.testing.assert_array_equal(first['alpha_mask'], second['alpha_mask'])
    np.testing.assert_array_equal(first['painted_images'], second['painted_images'])
