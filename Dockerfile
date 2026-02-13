# 1. Use the dolfinx image
FROM dolfinx/lab:stable

# 2. Binder Compatibility Setup (REQUIRED)
# Binder forces the user to be 'jovyan' (UID 1000). 
# Without this block, your build will fail with permission errors.
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

USER root

# Create the user Binder needs
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# 3. Copy your repo files into the image and fix permissions
COPY . ${HOME}
RUN chown -R ${NB_UID} ${HOME}

# 4. Switch to the Binder user
USER ${NB_USER}
WORKDIR ${HOME}

# 5. Install your requirements
RUN pip install --no-cache-dir -r requirements.txt
