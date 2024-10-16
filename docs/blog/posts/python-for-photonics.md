---
date: 2024-10-10
authors: [dusan]
---

# Python for Photonics Development

[Python](https://www.python.org/) has become an indispensable tool in many technical fields, including ours. It gives researchers and engineers a versatile platform for simulating, modeling, and analyzing photonic systems, effectively streamlining the **entire development workflow.** With its extensive ecosystem of libraries and tools, Python enables efficient and innovative solutions in photonics that we think are worth exploring.

In this guide, we'll walk you through setting up your Python development environment, introduce essential libraries, explore cloud and remote tools to leverage computational resources, share valuable settings for an enhanced coding experience, and provide additional tips to elevate your photonics projects using Python.

<!-- more -->

!!! note "Note"

    PreFab is developed using Python, and we leverage many Python packages to enhance our development process. This guide is a compilation of resources and insights that have been beneficial in *our* work. While it's not exhaustive, we hope it proves useful to you in your photonics development endeavors.

## Essential Python Libraries for Photonics

The Python ecosystem offers a variety of powerful tools specifically designed for photonics design and development. Here are some of the most notable ones:

- **[Tidy3D](https://www.flexcompute.com/tidy3d/)**: A commercial cloud-based FDTD solver for simulating electromagnetic fields in photonic structures, offering ultrahigh-speed simulations, an intuitive Python API, and *incredible* documentation and tutorials.

- **[Meep](https://meep.readthedocs.io/en/latest/)**: An open-source software package for simulating electromagnetic systems using FDTD, with extensive Python bindings for scripting and automation. Meep also has fantastic documentation and tutorials to help you learn not just how to use the software, but also fundamentals of photonics and electromagnetics.

- **[gdsfactory](https://gdsfactory.github.io/gdsfactory/)**: An industry-leading, open-source Python library for designing and automating photonic integrated circuits, facilitating layout generation, verification, and integration with simulation tools. If you are designing photonic integrated circuits with Python, this is a fantastic starting point.

- **[KLayout](https://www.klayout.org/klayout-pypi/):** KLayout is a versatile 2D viewer and editor for mask layouts in chip manufacturing. The KLayout Python module extends its functionality, allowing for scripting and automation within the KLayout system.

- **[gdstk](https://heitzmann.github.io/gdstk/)**: A Python library for creating and manipulating GDS files, supporting complex geometric operations essential for photonic design.

- **[Awesome Photonics](https://github.com/joamatab/awesome_photonics)**: A curated GitHub repository compiling a comprehensive list of photonics tools and resources, many of which are Python-based or compatible with a Python-based workflow. This repository is an excellent starting point to discover new libraries, frameworks, and tools in the photonics domain.

## Setting up Your Integrated Development Environment (IDE)

Choosing the right Integrated Development Environment (IDE) is crucial for an efficient and productive workflow in photonics development. Here are three excellent options to consider:

- **[Visual Studio Code (VS Code)](https://code.visualstudio.com/)**: Visual Studio Code is a free, open-source IDE developed by Microsoft. It's highly popular due to its versatility (much like Python) and extensive extension marketplace. VS Code offers powerful features like IntelliSense (smart code completion), debugging tools, Git integration, and customizable workspaces, making it an excellent choice for Python development.

- **[Cursor](https://cursor.sh/)**: Cursor is an enhanced version (fork) of VS Code, augmented with thoughtful AI tools that assist in code development. With integrated AI capabilities, Cursor can help you write, debug, and optimize your code more efficiently. It provides features like code autocompletion, intelligent code analysis, and automated code refactoring, making it a powerful tool for both beginners and experienced developers.

- **[Zed](https://zed.dev/)**: Zed is a new, ultra-fast IDE that focuses on performance and a responsive coding experience. While it is still under active development and may have limited functionality and operating system support, Zed aims to provide a sleek and efficient environment for developers. Its emphasis on speed and minimalism could make it a strong contender as it matures. We recommend keeping a close eye on this IDE as it evolves.

For the rest of this guide, we will use VS Code (and therefore Cursor).

### Essential VS Code Extensions for Python Development

To further enhance your development experience in VS Code (or Cursor), add these essential extensions (see below for how to install them):

- **[Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)**: The Python extension for VS Code provides rich support for the Python language, including features like IntelliSense, linting, debugging, code navigation, code formatting, and testing. It is maintained by Microsoft and is one of the most installed extensions for Python development.

- **[Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)**: The Jupyter extension for VS Code allows you to work with Jupyter Notebooks directly within VS Code, offering an interactive environment for data analysis, visualization, and computational experimentation. We love Jupyter for rapidly prototyping and exploring new ideas, and this extension makes working with them in VS Code a pleasure.

- **[Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)**: The Ruff extension for VS Code is a fast, highly configurable linter (code quality check) and formatter for Python, designed to help you maintain code quality and adhere to coding standards. It is so fast that it can run in the background while you code, and it will notify you of any issues in your code as you save your file (which we will configure later in this guide). We also love Ruff because it is opinionated and enforces a consistent style guide, which can help you focus on what really matters and not waste precious development time on style choices.

- **[GitHub Theme](https://marketplace.visualstudio.com/items?itemName=GitHub.github-vscode-theme)**: The GitHub Theme (Dark) for VS Code is a visually appealing theme that enhances the appearance of your editor, making it easier on the eyes (especially in the evening) and more enjoyable to work with. Even if you don't like this theme, we think it's a good idea to browse the marketplace and find a theme that you do like, making for a more pleasant coding experience under long hours of work.

#### How to Install Extensions

1. **Open VS Code**: Launch Visual Studio Code on your computer.

2. **Access the Extensions View**: Click on the Extensions icon in the Activity Bar on the side of the window or press `Ctrl+Shift+X` (`Cmd+Shift+X` on macOS).

3. **Search for Extensions**: In the Extensions view, type the name of the extension you want to install (e.g., "Python", "Jupyter", "Ruff") in the search bar.

4. **Install the Extension**: Click the `Install` button next to the extension in the search results.

5. **Reload if Necessary**: Some extensions may require you to reload VS Code to activate them. If prompted, click the `Reload` button.

### Useful VS Code Settings

Optimizing your VS Code settings can enhance your coding efficiency and code quality. Below are some recommended settings tailored for Python development:

**Enable Format on Save for Python Files and Notebooks**: Automatically format your code every time you save a file, both in Python scripts and Jupyter Notebooks.

- Go to `File` > `Preferences` > `Settings` (or press `Ctrl+,` on Windows/Linux or `Cmd+,` on macOS).
- Search for `Format On Save`.
- Check the box for **`Editor: Format On Save`**.
- Search for `Notebook: Format On Save`.
- Check the box for **`Notebook: Format On Save`**.

**Set Ruff as the Default Formatter for Python and Notebooks**:

- Open your settings and search for `Editor: Default Formatter`.
- Select **`Ruff`** from the dropdown menu.
- Search for `Notebook: Default Formatter`.
- Set it to use **`Ruff`** for formatting code cells in Jupyter Notebooks.

!!! success "Success"
    Now, with just a click of the save button, Ruff will automatically enhance the appearance and readability of your code.

**Configure Jupyter Startup Commands**:

- These commands ensure that your notebook automatically reloads modules before executing code, which is helpful during development. This one has saved us many headaches in the early days of working with Python.
- Add the following setting to your `settings.json` file:

  ```json
  "jupyter.runStartupCommands": [
    "%load_ext autoreload",
    "%autoreload 2"
  ],
  ```

**Accessing the `settings.json` File**:

- To open the `settings.json` file in VS Code:
  - Go to `File` > `Preferences` > `Settings` (or press `Ctrl+,` on Windows/Linux or `Cmd+,` on macOS).
  - In the Settings page, click on the **Open Settings (JSON)** icon (a document with a small arrow) in the upper-right corner. This will open the `settings.json` file where you can edit your settings directly.
  - Alternatively, you can use the Command Palette:
    - Press `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS) to open the Command Palette.
    - Type `**Open User Settings (JSON)**` and select it from the list.

**Example `settings.json` Entries**:

Replace or update your `settings.json` with the following entries:

```json
{
    "editor": {
        "formatOnSave": true,
        "codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        },
        "defaultFormatter": "charliermarsh.ruff"
    },
    "notebook": {
        "formatOnSave": {
            "enabled": true
        },
        "codeActionsOnSave": {
            "notebook.source.fixAll": true,
            "notebook.source.organizeImports": true
        },
        "defaultFormatter": "charliermarsh.ruff"
    },
    "jupyter.runStartupCommands": [
        "%load_ext autoreload",
        "%autoreload 2"
    ],
}
```

!!! tip "Thought"
    Customizing your IDE to suit your workflow—including setting up keyboard shortcuts that work best for you—can significantly improve your productivity and make your coding experience more enjoyable.

## Setting Up Your Python Virtual Environment

Creating a Python virtual environment is essential for managing dependencies and ensuring a clean development setup. It allows you to isolate your project's packages and Python version from other projects and the system-wide Python installation.

> We think it's a good idea to use a virtual environment manager to keep your global Python installation clean and to avoid conflicts between different projects.

Here are three popular options for managing virtual environments:

- **[Conda](https://docs.conda.io/en/latest/)**: Conda is a cross-platform package and environment manager that can handle both Python and non-Python dependencies, making it ideal for complex projects in scientific computing and data science. It supports creating isolated environments and includes its own package repositories.

- **[venv](https://docs.python.org/3/library/venv.html)**: The venv module is the built-in tool for creating virtual environments in Python 3. It's simple to use and comes pre-installed with Python, making it a convenient choice for managing environments with different packages or Python versions.

- **[uv](https://astral.sh/uv/)**: uv is a high-performance virtual environment manager created by Astral (the creators of Ruff). It offers rapid environment creation and management, aiming to be a minimalistic and efficient alternative to existing tools. Although still in its early stages, uv is designed to push the boundaries of performance and development ergonomics. We get inspired just by using it, and we think you will too.

For the rest of this tutorial, we will use Conda due to its popularity and robust feature set.

### Setting Up Conda

To set up a Python virtual environment using Conda, follow these steps:

**1. Install Conda**

If you don't have Conda installed, you can install either Anaconda or Miniconda. Miniconda is a minimal installer for Conda that includes only Conda and its dependencies.

- **Download Miniconda** from the [official website](https://docs.conda.io/en/latest/miniconda.html) suitable for your operating system.

- **Install Miniconda** by running the installer and following the on-screen instructions.

**2. Create a New Environment**

Open the integrated terminal in VS Code by going to `View` > `Terminal` (or pressing `` Ctrl+` `` on Windows/Linux or `` Cmd+` `` on macOS). Use the following command to create a new Conda environment. Replace `myenv` with your desired environment name and `3.11` with the Python version you need.

```bash
conda create -n myenv python=3.11
```

**3. Activate the Environment**

Activate your new environment with the following command:

```bash
conda activate myenv
```

**4. Install Necessary Packages**

You can now install any packages you need using Conda or `pip`. For example, to install NumPy and SciPy, use:

```bash
conda install numpy scipy
```

Or, to install a package via `pip`:

```bash
pip install prefab
```

### Selecting the Conda Environment in VS Code

After setting up your Conda environment, you'll want to ensure that VS Code uses this environment for running and debugging your Python code and Jupyter Notebooks.

#### Choosing the Python Interpreter

**1. Open a Python File**:

- Open any Python file in VS Code or create a new one with the `.py` extension.

**2. Select the Interpreter**:

- Click on the Python version displayed in the bottom-right corner of the VS Code status bar. It might say something like `Python 3.x.x`.

- A list of available Python interpreters will appear. Look for your Conda environment in the list. It should be listed with the name you gave it (e.g., `myenv`) and the path to the Python executable.

- Select your Conda environment from the list.

**3. Verify the Interpreter**:

- Once selected, the status bar should update to reflect the interpreter from your Conda environment.

#### Choosing the Jupyter Kernel

When working with Jupyter Notebooks in VS Code, you need to select the appropriate kernel associated with your Conda environment.

**1. Open a Jupyter Notebook**:

- Open an existing `.ipynb` file or create a new one in VS Code.

**2. Select the Kernel**:

- At the top-right corner of the notebook editor, you will see the kernel name (e.g., `Python 3`). Click on it to change the kernel.

- In the kernel selection dialog, you should see a list of available kernels. Your Conda environment should be listed, often with the name of the environment and the Python version.

- Select the kernel corresponding to your Conda environment.

**3. Verify the Kernel**:

- The kernel name should update to reflect the selected Conda environment.

**Troubleshooting**:

If your Conda environment's kernel doesn't appear in the list, you may need to install the `ipykernel` package in your environment:

```bash
conda activate myenv  # or the name of your environment
conda install ipykernel
```

Alternatively, you can register the environment's kernel manually:

```bash
python -m ipykernel install --user --name=myenv
```

After installing the kernel, restart VS Code, and it should recognize the new kernel associated with your Conda environment.

By selecting the correct Python interpreter and Jupyter kernel, you ensure that your code runs in the intended environment, using the packages and dependencies you've installed.

!!! info "Info"
    And that's it! The rest of this guide will focus on additional tools and resources that can help you take your Python-based photonics development to the next level.

## Leveraging Cloud and Remote Tools for Computational Resources

Photonics simulations and computations can be resource-intensive, often requiring significant computational power and specialized hardware. Leveraging cloud and remote tools allows you to access powerful computational resources without the need for expensive local hardware. Here are some platforms and tools that can help you take advantage of cloud computing:

- [Google Colab](https://colab.research.google.com/) is a free cloud service (with some paid options) provided by Google that enables you to write and execute Python code through a web browser. It offers free access to GPUs and TPUs, allowing you to accelerate your simulations and computations using powerful hardware.

- [GitHub Codespaces](https://github.com/features/codespaces) provides a full-fledged cloud-based development environment directly within GitHub. It allows you to develop in the cloud using a cloud-hosted VS Code environment that's configured for your project.

- [Modal](https://modal.com/) is a platform that allows you to run your code in the cloud with minimal setup. It provides serverless computing, enabling you to run functions and scripts without managing servers. We love Modal for how easy it is to leverage powerful compute without having to leave the local Python environment.

- [Lambda Labs](https://lambdalabs.com/) provides cloud GPU workstations and servers for deep learning and computational tasks. They offer powerful NVIDIA GPUs (though sometimes limited based on usage) and pre-configured environments with popular frameworks, saving setup time.

- [SSH VS Code Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) allows you to develop remotely by connecting to remote servers in your organization or cloud instances (e.g., AWS, GCP, Lambda Labs) directly from your local VS Code editor.

## Terminal Tools

In addition to a good IDE, having the right terminal tools can offer nice benefits. Here are some useful command-line tools you may want to install:

- [Git](https://git-scm.com/) is a distributed version control system that allows you to track changes in your code, collaborate with others, and manage different versions of your project. It's an indispensable tool for any developer, and we think it is an underrated tool for photonics development too. Although VS Code has a built-in Git client, using Git in the terminal can be more flexible. We recommend [this video](https://www.youtube.com/watch?v=K6Q31YkorUE) for a quick introduction on using Git.

- [htop](https://github.com/htop-dev/htop) is an interactive process viewer for Linux and macOS systems. It provides a real-time view of system processes, CPU, and memory usage, making it especially easy to monitor remote compute instances.

- [nvtop](https://github.com/Syllo/nvtop) is like `htop` for NVIDIA GPUs. There are also many other `top`-like tools out there, so pick the one that best fits your system and workflow.

- [Oh My Zsh](https://ohmyz.sh/) is a popular shell framework that enhances the capabilities of the Zsh shell (which we also recommend using). It provides a rich set of plugins that make using the terminal much more powerful. If you find yourself using the terminal even just a little bit, this is a great tool to have.

## AI Coding Assistants

AI-powered coding assistants have become invaluable tools for developers, whether you're just starting with Python or exploring new concepts in any field. These assistants offer real-time help, explanations, and code suggestions, making them excellent learning aids and productivity boosters.

- **[ChatGPT](https://chat.openai.com/)**: Developed by OpenAI, ChatGPT is *the* widely recognized (large) language model. They make improvements to their models and user interface regularly, so we recommend staying updated with its latest features to ensure you're utilizing the most advanced tools available.

- **[Claude](https://www.anthropic.com/)**: Claude, developed by Anthropic, is a competitor to ChatGPT. While benchmarking language models can be challenging (many believe Claude offers superior capabilities at least in some contexts), using multiple models from different providers (there are many others!) can help you obtain the best possible answers.

- **[Cursor](https://cursor.sh/)**: As previously mentioned, Cursor is an AI-enhanced IDE based on VS Code. One of its unique features is the ability to choose from multiple language models, helping you get the most accurate and helpful responses at any given time.

- **[GitHub Copilot](https://github.com/features/copilot)**: GitHub Copilot integrates seamlessly into VS Code, providing AI-assisted code completion and suggestions. While we believe Cursor offers superior features as an AI coding assistant, Copilot is a strong option if you prefer VS Code as your IDE.

There has been considerable discussion surrounding AI assistants, with both positive and negative viewpoints. However, when used pragmatically, these tools can [enhance productivity](https://github.blog/news-insights/research/research-quantifying-github-copilots-impact-on-developer-productivity-and-happiness/). If you're not already using these tools, we strongly encourage you to consider doing so.

## Other Resources

While AI coding assistants are a great starting point when you need help with code, there are numerous online communities and resources available to assist with more specialized problems that other developers may have already solved. Engaging with these communities can provide valuable insights and support.

- **[Stack Overflow](https://stackoverflow.com/)**: A premier platform for getting help with specific coding issues. You can ask questions, share knowledge, and learn from the experiences of other developers.

- **[GitHub](https://github.com/)**: Many photonics projects and libraries are hosted on GitHub. By engaging with repositories' issues and discussions, you can get direct support from developers and other users. Contributing to open-source projects or raising issues can also enhance your understanding and help improve the tools you use.

- **"Think Python: How to Think Like a Computer Scientist" by Allen B. Downey**: An excellent introduction to Python programming, focusing on developing a computational thinking mindset. Python is an evolving language, so older books might not cover everything you need. But this one is so elegantly organized and written that it's still a great resource.

- **"Effective Computation in Physics: Field Guide to Research with Python" by Anthony Scopatz and Kathryn D. Huff**: This guidebook covers Python in the context of scientific computing and physics research, making it particularly useful for photonics developers.

- **"Talk Python to Me"**: A podcast that explores various topics in the Python ecosystem through interviews with experts and contributors. There is more to a language than just the syntax and libraries, and this podcast offers a great perspective on the language from the people who help shape it.

## What's Next

To further advance your photonics development journey with Python:

1. **Explore Photonics Simulations**: Experiment with examples from [Meep](https://meep.readthedocs.io/en/latest/) and [Tidy3D](https://docs.flexcompute.com/projects/tidy3d/en/latest/) documentation to deepen your understanding of photonic simulations and computational electromagnetics.

2. **Contribute to Open-Source Projects**: Get involved with projects like [gdsfactory](https://github.com/gdsfactory/gdsfactory) and [SiEPIC](https://github.com/lukasc-ubc/SiEPIC-Tools). Contributing allows you to collaborate with others, improve existing tools, and engage with the photonics community.

3. **Develop Your Own Tools!**: We can't wait to see what you build.

And of course, you can run our [notebook examples](../../examples/1_prediction.ipynb) on using virtual nanofabrication models to add fabrication-awareness to your photonics simulations and designs.
