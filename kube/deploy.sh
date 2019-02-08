#!/usr/bin/env bash

#Installing Weave for networking
export kubever=$(kubectl version | base64 | tr -d '\n')
kubectl apply -f "https://cloud.weave.works/k8s/net?k8s-version=$kubever"

#MetalLB for bare metal loadbalancer
kubectl apply -f https://raw.githubusercontent.com/google/metallb/v0.7.3/manifests/metallb.yaml
#Configure the IP pool
kubectl apply -f metallb-config.yaml

#Creating the service and the deployment of the tensorflow serving container
kubectl apply -f mnist-service.yaml
kubectl apply -f mnist2.yaml
