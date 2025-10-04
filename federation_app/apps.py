from django.apps import AppConfig

class FederationAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'federation_app'
    verbose_name = '联邦学习应用'