echo "What do you want to do?"
echo "1. Train a network using test_mnist"
echo "2. Only executes MNIST_MCSim "
echo "3. Train+MCsim "
echo "option: " 
read option
case $option in
	1)
		echo "Start training the network"
		echo "******************************************************************************"
		python3 test_mnist.py
	;;
	2)
		echo "Starting a Monte Carlo simulation"
		echo "******************************************************************************"
		python3 MNIST_MCSim.py
	;;
	3)
		echo "Start training the network"
		echo "******************************************************************************"
		python3 test_mnist.py
		sleep 1
		echo "******************************************************************************"
		echo "Starting a Monte Carlo simulation"
		echo "******************************************************************************"
		python3 MNIST_MCSim.py
	;;
	*)
		echo "Wrong option"
	;;
	esac
