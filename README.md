# 3kings_long_short_v3


### Installation

---
1. Clone the repository

   - Using SSH
   ```shell script
   git clone xxx
   ``` 
   
   - Using HTTPS with Personal Access Token
   ```shell script
   git clone xxx
   ```

2. Set up the Virtual Environment

    Ubuntu 20.04 (Debian-based Linux)
    ```shell script
    cd ./algo_3kings_long_short_v3
    python3.12 -m venv venv/
    source ./venv/bin/activate
    ```
   
    Windows 10
    ```shell script
    cd .\algo_3kings_long_short_v3
    python -m venv .\venv\
    .\venv\Scripts\activate
    ```

3. Install the dependencies

    ```shell script
    pip install -r requirements.txt
    pip install --upgrade pip
    ```


### Deployment

---
#### Dev Environment
1. Run the application
    ```shell script
    python3.12 simulate_trading.py
    ```

#### Running via Systemd
1. Move the file to Systemd's system folder.
    ```shell script
    sudo cp ./sim_long_short.service /etc/systemd/system/sim_long_short.service
    ```
2. Enable and start the service.
    ```shell script
    sudo systemctl daemon-reload
    sudo systemctl enable sim_long_short.service
    sudo systemctl start sim_long_short.service
    ```
3. Check if the application is running.
    ```shell script
    sudo systemctl status sim_long_short.service
    ```
# 3kings_long_short_v3

