import registerRootComponent from 'expo/build/launch/registerRootComponent';
import AsyncStorage from '@react-native-async-storage/async-storage';
import App from './App';
import { Amplify } from 'aws-amplify'
import awsconfig from './src/aws-exports'

Amplify.configure(awsconfig)

registerRootComponent(App);
