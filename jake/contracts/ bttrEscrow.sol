pragma solidity ^0.5.0;

contract bttrEscrow {

    // mapping msg.sender --> address, and msg.value --> bet/Eth
    // mapping(address => address(msg.sender), ) public bettingInfo;

    // event to establish changing of odds
    event Odds(int currentOdds);

    //Function where user will submit their bets
    function makeBet(uint betAmount) public payable {
        address(this).transfer(betAmount);
    }

    //gotta write the function that can do the math and call the function
    //can make a function that writes to the mapping, python can call that function
    // Payout function 
    function payout(address payable recipient, uint amount) public payable{
        if (amount < address(this).balance) {
            recipient.transfer(amount);
            //uint balance = address(this).balance;
        }
        // recalling all the bets made for this instance

        // calculate payouts based on odds for that instance

        // for each address, payout --> loop

        // delete existing bets for current pitch

        // How do you trigger a private function?
        // How do you refer to the model's odds and pull the values into the contracts?

    }

    // Fallback function
    function() external payable{}

    // CHECK FUNCTIONS
    // function checkBalance() public returns (uint){
    //     return address(this).balance;
    // }

    // function checkMessage() public returns (string memory){
    //     return msg.data;
    // }

}