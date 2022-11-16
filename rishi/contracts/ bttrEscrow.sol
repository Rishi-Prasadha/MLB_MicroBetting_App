pragma solidity ^0.5.0;

import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/release-v2.5.0/contracts/token/ERC721/ERC721Full.sol";
import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/release-v2.5.0/contracts/crowdsale/Crowdsale.sol";
import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/release-v2.5.0/contracts/crowdsale/emission/MintedCrowdsale.sol";
import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/release-v2.5.0/contracts/token/ERC20/ERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/release-v2.5.0/contracts/token/ERC20/ERC20Detailed.sol";
import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/release-v2.5.0/contracts/token/ERC20/ERC20Mintable.sol";

contract bttrEscrow is ERC20, ERC20Detailed, ERC20Mintable, Crowdsale, ERC721Full, MintedCrowdsale{

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