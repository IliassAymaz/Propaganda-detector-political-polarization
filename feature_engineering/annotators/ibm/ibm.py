import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_cloud_sdk_core.api_exception import ApiException
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions

class IBMAnnotator:
    def __init__(self, api_key, service_url):
        authenticator = IAMAuthenticator('oKhS6x160MtEXxNo504Jul7DeArTSN9XaSmoTka68VOP')
        self.nlu = NaturalLanguageUnderstandingV1(
                    version='2019-07-12',
                        authenticator=authenticator
                        )
        self.nlu.set_service_url('https://gateway-lon.watsonplatform.net/natural-language-understanding/api')

    def annotateEmotions(self,text):
        try:
            response = self.nlu.analyze(
                    text=text,
                    features=Features(emotion=EmotionOptions(document=True))).get_result()
            return response['emotion']['document']['emotion']
        except ApiException:
            return {
                   'sadness' : 0,
                   'joy' : 0,
                   'fear' : 0,
                   'disgust' : 0,
                   'anger' : 0
                    }
