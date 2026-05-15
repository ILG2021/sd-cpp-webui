# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['../app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('..\\.venv\\Lib\\site-packages\\gradio', 'gradio/'),
    	('..\\.venv\\Lib\\site-packages\\gradio_client', 'gradio_client/'),
    	('..\\.venv\\Lib\\site-packages\\safehttpx', 'safehttpx/'),
    	('..\\.venv\\Lib\\site-packages\\groovy', 'groovy/'),
    	('..\\.venv\\Lib\\site-packages\\stable_diffusion_cpp\\lib', 'stable_diffusion_cpp/lib'),
    	('../config', 'config'),
        ('../.env', '.env'),
    	],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    module_collection_mode={
        'gradio': 'py',
    },
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='imagekit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='imagekit',
)

