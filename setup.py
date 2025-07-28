from setuptools import setup, find_packages

setup(
    name="core-dataset-lib",        # имя вашей библиотеки
    version="0.1.0",                 # текущая версия
    packages=find_packages(),        # автоматически найдёт все пакеты в проекте
    install_requires=[               # список зависимостей, если есть
        # "requests>=2.25.0",
    ],
    author="litgax, spectrespect",
    author_email="emaillit8@gmail.com, emailgax8@gmail.com",
    description="Library for convenient work with datasets",
    url="https://github.com/KoTeuKaSeeker/corelib",  # ссылка на репозиторий
    classifiers=[                    # метаданные для PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",         # минимальная поддерживаемая версия Python
)