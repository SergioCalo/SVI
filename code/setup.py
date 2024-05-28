from setuptools import setup, find_packages


def main():
    setup(
        name='sinkhorn',
        version='0.0.1',
        author="Sergio Calo et al.",
        packages=find_packages('sinkhorn'),
        package_dir={'': 'sinkhorn'},
        setup_requires=['wheel'],
    )


if __name__ == "__main__":
    main()
