import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyD_WovNYJmKwJH-4jeX84FL7bJ2fkgJJWg",
  authDomain: "price-optimiser-auth.firebaseapp.com",
  projectId: "price-optimiser-auth",
  storageBucket: "price-optimiser-auth.firebasestorage.app",
  messagingSenderId: "513371291536",
  appId: "1:513371291536:web:056b1f43c7147be750fd15",
  measurementId: "G-CLBF56TE0D"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and get a reference to the service
export const auth = getAuth(app);
