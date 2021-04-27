# PART 2
In part 1, we learnt how to train a follow-up question generation model (followQG) by fine-tuning GPT-2. In this part we will use that model to create a virtual interviewer using AWS Sumerian.

## Workflow
We will see a demo of how to build a virtual 3D animated host as an interviewer. The basic workflow of the application is as follows:
- Animated interviewer asks a pre-decided question to the candidate
- Candidate answers the question posed by the host
- Through the candidate’s device microphone their response is recorded and processed to text format (speech-to-text)
- The processed response is sent to follow up generator model
- This model generates a follow up question based on the candidate’s response to the base question (trained from Part 1)
- And the generated follow-up question is in return asked by the interviewer host
- After user gives his response, host picks a new question from the pre-defined set and the cycle continues

## Follow-up Question Generator API Setup
Deploy the trained followQG model on a server (for eg. on a AWS EC2 instance). All the required scripts for that is in [part-2](part-2)

### Install
```bash
pip install -r requirements.txt
```

Deploy it as a Flask app following this [blog](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04).

### Run the app
```bash
$ gunicorn app:app
```

## AWS Sumerian Setup
In order to replicate the entire interview set up first we need to create an AWS account (click [here](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/) to know the procedure to activate an AWS account).

1. After creating an AWS account, sign in to the console and select the Amazon Sumerian Service. Once Sumerian is loaded, create a new scene.
2. Import already built scene as a [bundle](part-2/assets/virtual_interviewer_v1-gltf-20210425_162917.zip) in the form of an asset and recreate the entire environment.
3. Creating Cognito Identity Pool ID
    1. Open Cognito Console or click this [link](https://us-west-2.console.aws.amazon.com/cognito/home?region=us-west-2).
    2. Choose Manage Identity Pools and click on Create new identity pool.
    3. Enter a name for the identity pool and remember this name.
    4. Enable Access to unauthenticated identities(to enable the services to be used when its published for guest users) and Allow basic flow.
    5. Click on Create Pool and on Allow in the next page to allow the identity pool to be used by other users or roles ,called as [IAM roles](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html).
    6. Copy the Identity Pool ID or store it securely.
4. Configuring Sumerian Scene with Cognito Identity Pool ID and adding Polly permissions
    1. Open Identity and Access Management (IAM) Dashboard and click on roles.
    2. Select the Unauthorized Pool role of the created Identity Pool ID of the format Console `Your Pool ID name` Role.
    3. Add Inline Policy and choose service as Polly, and add action as SynthesizeSpeech.
    4. Select All Resources and click on Review policy.
    5. Enter a name for the policy and create policy.
5. Call the followQG API 
Replace the <your_server_domain_name> in the UserSpeech script in Sumerian scene you created.

You now run the scene and publish it. Have fun!


## References
1. [Sumerian Youtube Tutorials](https://www.youtube.com/watch?v=J3zsG0ejgO8&list=PLhr1KZpdzukd0g3qrxrCzwfZF97Ylprpy)
2. [Speech and Host Component](https://docs.sumerian.amazonaws.com/tutorials/create/beginner/host-speech-component/)
3. [SpeechRecognition Documentation](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition)
4. [Sumerian Host Tutorials](https://docs.sumerian.amazonaws.com/courses/host-course//)
5. [Sumerian Scene Creation](https://docs.sumerian.amazonaws.com/tutorials/create/getting-started/light-switch/)
6. [Create Identity Pool](https://docs.aws.amazon.com/cognito/latest/developerguide/tutorial-create-identity-pool.html/)

I would like to acknowledge the help from Manish, Siddharth, Nikhil and Nipun on the AWS Sumerian setup.



