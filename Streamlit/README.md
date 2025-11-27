# Learning Streamlit
In the Pythonic realm 🐍👑 of data and machine learning web application frameworks, Streamlit has been acquiring quite a lot of popularity. In fact, Streamlit is designed to be incredibly user-friendly. You can create interactive web apps with just a few lines of Python code. It abstracts away much of the complexity involved in web development, making it accessible even to those without a web development background. 

__Please fork this repo and proceed with the installation.__

In this document you will find:
- [Installation steps](#installation)
- [Streamlit essentials](#streamlit-essentials)
- [More resources](#further-readings)  
</br>

<a id="installation"></a>
## 💻 Installation

For __MacOs__/__Linux__ users

```bash
pyenv local 3.11.3 
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For __Windows__ users with __PowerShell CLI__

```Powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For __Windows__ users with __GIT-BASH CLI__

```bash
pyenv local 3.11.3 
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 🧪 Test Installation
You can check if Streamlit is correctly installed by running a sample app with the following command:
```bash
streamlit hello
```
This should open a Streamlit web app in your default web browser.  
</br>

<a id="streamlit-essentials"></a>
## 📊 Streamlit Essentials

The following  is based on the official [Streamlit documentation source](https://docs.streamlit.io/library/get-started/main-concepts).

- Streamlit is a Python library for creating interactive web apps for data science and machine learning.
- You build apps by adding Streamlit commands to a Python script and running it with `streamlit run my_app.py`.

### Development Flow

- Streamlit provides an interactive development loop: Write code, save, view results, and iterate.
- Changes in your script trigger automatic updates in the app.
- For a smooth development experience, arrange your code editor and app preview side by side.

### Data Flow

- Streamlit re-runs your entire script whenever changes occur in the source code or when users interact with app widgets.
- Widgets are elements like sliders and buttons that enable user interaction.
- Use the `@st.cache_data` decorator to optimize performance by skipping costly computations.

### Display and Style Data

- Streamlit supports displaying data in various ways, including text, tables, and charts.
- You can use methods like `st.write()`, `st.dataframe()`, and `st.table()`.
- Customization options are available for styling data frames.

### Widgets

- Widgets are used to capture user input and display results.
- Examples include:
  - Slider: `weight = st.slider('Select weight', min_value=0, max_value=1000, step=0.5)`
  - Button: `st.button('Press me!')`
  - Select Box: `model = st.selectbox(label="Select an model", options=['LogReg','Random Forest', 'XGBoost']`

### Layout

- You can organize widgets and content in a sidebar using `st.sidebar`.
- `st.columns()` allows widgets to be displayed side by side.
- Use `st.expander` to hide or show content and save space.

### Progress and Themes

- Show progress with `st.progress()` for long-running computations.
- Streamlit supports light and dark themes, which you can customize in the settings.

### Caching

- Streamlit provides caching decorators to store and reuse the results of expensive function calls.
- Use `@st.cache_data` for computations that return data and `@st.cache_resource` for global resources.

### Multipage Apps

- Organize large apps into multiple pages for easier management and navigation.
- Create separate Python script files for each page and place them in a "pages" folder.
- Each script corresponds to a different page in the app.

### App Model

- Streamlit apps are Python scripts that execute from top to bottom.
- User interactions and changes trigger script reruns.
- Caching is used to optimize performance and speed up app responses.  
</br>

<a id="further-readings"></a>
## 📚 Further Readings

The following is a list of other popular python web frameworks:
+ [Django](https://www.djangoproject.com)
+ [Dash](https://dash.plotly.com)
+ [Flask](https://flask.palletsprojects.com/en/3.0.x/)
+ [FastAPI](https://fastapi.tiangolo.com)
+ [Taipy](https://docs.taipy.io)  

Both FastAPI and Flask are primarily focused on backend development for web applications.  
</br>

## ⚖️ License
[MIT license](LICENSE)
