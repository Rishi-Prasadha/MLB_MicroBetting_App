pragma solidity ^0.5.0;

contract bttrEscrow {

    //Function where user will submit their bets
    function makeBet(uint betAmount) public payable {
        // address(this).send(betAmount)
    }
    
    function makeBet2() public payable {

    }
   
    // Payout function 
    function payout(address payable recipient, uint amount) public payable{
        if (amount < address(this).balance) {
            recipient.transfer(amount);
            //contractBalance = account(this).balance;
            //uint balance = address(this).balance;
        }

        // How do you trigger a private function?
        // How do you refer to the model's odds and pull the values into the contracts?

    }

    function checkBalance() public view returns (uint) {
        return address(this).balance;
    }

    // Fallback function
    function() external payable{}

}