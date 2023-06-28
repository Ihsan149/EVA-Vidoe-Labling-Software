from django.core.management.base import BaseCommand
from annotator.kcftracker import compile_fhog


class Command(BaseCommand):
    help = 'Compiles the fhog_utils file to be used in cpp through numba'

    def handle(self, *args, **options):
        compile_fhog()
