
# -> base image with Python 3.9
FROM python:3.9

# -> set working directory inside container
WORKDIR /app

# -> copy all deployment files into the container
COPY . .

# -> install all dependencies
RUN pip3 install -r requirements.txt

# -> create a non-root user as required by HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# -> run the Streamlit app on port 7860 -- required by HF Spaces
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
