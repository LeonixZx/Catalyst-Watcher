from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = collect_all('streamlit')

hiddenimports += collect_submodules('streamlit')
hiddenimports += [
    'streamlit.web.cli',
    'streamlit.runtime.scriptrunner.magic_funcs',
    'streamlit.runtime.caching',
    'importlib_metadata',
    'streamlit.config',
    'pyngrok',
    'langchain',
]