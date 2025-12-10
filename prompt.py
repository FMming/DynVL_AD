REAL_NAME = {'Brain': 'Brain', 'Liver':'Liver', 'Retina_RESC':'retinal OCT', 'Chest':'Chest X-ray film', 'Retina_OCT2017':'retinal OCT', 'Histopathology':'histopathological image'}

# Original ensemble prompt templates retained for ablation
TEMPLATES = [
    'a cropped photo of the {}.',
    'a cropped photo of a {}.',
    'a close-up photo of a {}.',
    'a close-up photo of the {}.',
    'a bright photo of a {}.',
    'a bright photo of the {}.',
    'a dark photo of the {}.',
    'a dark photo of a {}.',
    'a jpeg corrupted photo of a {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of the {}.',
    'a blurry photo of a {}.',
    'a photo of the {}',
    'a photo of a {}',
    'a photo of a small {}',
    'a photo of the small {}',
    'a photo of a large {}',
    'a photo of the large {}',
    'a photo of the {} for visual inspection.',
    'a photo of a {} for visual inspection.',
    'a photo of the {} for anomaly detection.',
    'a photo of a {} for anomaly detection.'
]

# Dynamic prompt rule library: modality-task keyed templates
DYNAMIC_PROMPT_LIB = {
    ('Brain', 'AC'): {
        'org': 'brain',
        'normal': [
            'a FLAIR MRI scan of [c] [org] (clear for classification)',
            'a T1-weighted MRI scan of [c] [org] (clear for classification)',
            'a T2-weighted MRI scan of [c] [org] (clear for classification)'
        ],
        'abnormal': [
            'a FLAIR MRI scan of [c] [org] tumor (distinct for classification)',
            'a T1-weighted MRI scan of [c] [org] lesion (distinct for classification)',
            'a T2-weighted MRI scan of [c] [org] edema (distinct for classification)'
        ],
    },
    ('Brain', 'AS'): {
        'org': 'brain',
        'normal': [
            'a FLAIR MRI scan of [c] [org] (detailed for segmentation)',
            'a T2-weighted MRI scan of [c] [org] (detailed for segmentation)',
            'a T1-weighted MRI scan of [c] [org] (detailed for segmentation)'
        ],
        'abnormal': [
            'a FLAIR MRI scan of [c] [org] lesion (detailed for segmentation)',
            'a T2-weighted MRI scan of [c] [org] tumor region (detailed for segmentation)',
            'a T2-weighted MRI scan of [c] [org] edema region (detailed for segmentation)'
        ],
    },
    ('Liver', 'AC'): {
        'org': 'liver',
        'normal': [
            'a contrast-enhanced CT scan of [c] [org] (clear for classification)',
            'a cross-sectional CT scan of [c] [org] (clear for classification)',
            'an arterial phase CT scan of [c] [org] (clear for classification)',
            'a venous phase CT scan of [c] [org] (clear for classification)'
        ],
        'abnormal': [
            'a contrast-enhanced CT scan of [c] [org] lesion (distinct for classification)',
            'a cross-sectional CT scan of [c] [org] tumor (distinct for classification)',
            'an arterial phase CT scan of [c] [org] tumor (distinct for classification)',
            'a venous phase CT scan of [c] [org] lesion (distinct for classification)'
        ],
    },
    ('Liver', 'AS'): {
        'org': 'liver',
        'normal': [
            'a cross-sectional CT scan of [c] [org] (detailed for segmentation)',
            'a CT image of [c] [org] parenchyma (detailed for segmentation)',
            'a portal venous phase CT image of [c] [org] parenchyma (detailed for segmentation)'
        ],
        'abnormal': [
            'a cross-sectional CT scan of [c] [org] lesion (detailed for segmentation)',
            'a CT image of [c] [org] tumor region (detailed for segmentation)',
            'an arterial phase CT scan of [c] [org] lesion region (detailed for segmentation)'
        ],
    },
    ('Retina', 'AC'): {
        'org': 'retina macula',
        'normal': [
            'a retinal OCT image of [c] macula (clear for classification)',
            'a retinal OCT image of [c] fovea (clear for classification)'
        ],
        'abnormal': [
            'a retinal OCT image of [c] macular edema (distinct for classification)',
            'a retinal OCT image of [c] drusen (distinct for classification)'
        ]
    },
    ('Retina', 'AS'): {
        'org': 'retina macula',
        'normal': [
            'a retinal OCT image of [c] macula (detailed for segmentation)',
            'a retinal OCT image of [c] fovea (detailed for segmentation)'
        ],
        'abnormal': [
            'a retinal OCT image of [c] macular edema (detailed for segmentation)',
            'a retinal OCT image of [c] drusen region (detailed for segmentation)'
        ]
    },
    ('Chest', 'AC'): {
        'org': 'chest',
        'normal': [
            'a radiograph of [c] [org] (clear for classification)',
            'a chest X-ray of [c] lungs (clear for classification)',
            'a radiograph of [c] lung fields (clear for classification)'
        ],
        'abnormal': [
            'a radiograph of [c] [org] pathology (distinct for classification)',
            'a chest X-ray of [c] lung lesion (distinct for classification)',
            'a chest X-ray of [c] pulmonary opacity (distinct for classification)',
            'a chest X-ray of [c] consolidation (distinct for classification)'
        ],
    },
    ('Chest', 'AS'): {
        'org': 'chest',
        'normal': [
            'a chest X-ray of [c] lungs (detailed for segmentation)',
            'a radiograph of [c] lung fields (detailed for segmentation)'
        ],
        'abnormal': [
            'a chest X-ray of [c] lung pathology region (detailed for segmentation)',
            'a chest X-ray of [c] pulmonary opacity region (detailed for segmentation)',
            'a chest X-ray of [c] consolidation region (detailed for segmentation)'
        ],
    },
    ('Histopathology', 'AC'): {
        'org': 'lymph node',
        'normal': [
            'a stained histopathology slide of [c] lymph node (clear for classification)',
            'an H&E-stained histopathology slide of [c] [org] (clear for classification)'
        ],
        'abnormal': [
            'a stained histopathology slide of [c] lymph node metastasis (distinct for classification)',
            'an H&E-stained histopathology slide of [c] [org] high nuclei density (distinct for classification)'
        ]
    },
    ('Histopathology', 'AS'): {
        'org': 'lymph node',
        'normal': [
            'a stained histopathology slide of [c] lymph node (detailed for segmentation)',
            'an H&E-stained histopathology slide of [c] [org] (detailed for segmentation)'
        ],
        'abnormal': [
            'a stained histopathology slide of [c] lymph node metastasis (detailed for segmentation)',
            'a stained histopathology slide of [c] lymph node metastasis region (detailed for segmentation)'
        ]
    },
}


def get_modality_key(obj):
    if obj in ['Retina_OCT2017', 'Retina_RESC']:
        return 'Retina'
    elif obj == 'Chest':
        return 'Chest'
    elif obj == 'Histopathology':
        return 'Histopathology'
    elif obj == 'Brain':
        return 'Brain'
    elif obj == 'Liver':
        return 'Liver'
    else:
        return obj


def resolve_dynamic_templates(obj, task):
    mod = get_modality_key(obj)
    key = (mod, task)
    entry = DYNAMIC_PROMPT_LIB.get(key)
    if entry is None:
        normal = [f'a medical image of [c] {mod.lower()} (for {task.lower()})']
        abnormal = [f'a medical image of [c] {mod.lower()} lesion (for {task.lower()})']
        return normal, abnormal
    org = entry.get('org', mod.lower())
    normal = [t.replace('[org]', org) for t in entry['normal']]
    abnormal = [t.replace('[org]', org) for t in entry['abnormal']]
    return normal, abnormal

