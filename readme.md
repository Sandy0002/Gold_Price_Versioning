## About this project
In this we will extend gold price monitoring setup to apply concept of IAC and model versioning

### People looking to interact can use below link
**Note:** It takes 60 values in a list format to give prediction.<br>
[Click Here](https://gold-price-monitoring.onrender.com/docs#/default/predict_predict_post)


PROJECT ROOT/
│
├── src/               # your ETL + model training code
├── models/            # model artifacts (pickles, keras, etc.)
├── logs/              # runtime logs
├── src_logger/        # logging setup
├── Tests/             # test scripts
├── Dockerfile
├── render.yaml
├── requirements.txt
└── .github/           # CI/CD workflows


### Key things to look while saving files and building models
1. Compatible language version(Eg: python)
2. Compatible framework version(Eg: tensorflow)
3. Compatible model saving version(Eg: keras)
4. Adding right script in actions 