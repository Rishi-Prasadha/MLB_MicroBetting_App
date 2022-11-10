pragma solidity ^0.5.0;

import "./bttrToken.sol";
import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/release-v2.5.0/contracts/crowdsale/Crowdsale.sol";
import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/release-v2.5.0/contracts/crowdsale/emission/MintedCrowdsale.sol";

contract bttrTokenCrowdsale is Crowdsale, MintedCrowdsale{
    constructor(
        
        uint rate,
        address payable wallet,
        bttrToken token
    )

    Crowdsale(rate, wallet, token) public {}
}

contract bttrTokenCrowdsaleDeployer {

    address public bttrTokenAddress;
    address public bttrCrowdsaleAddress;

    constructor(

        string memory name,
        string memory symbol, 
        address payable wallet

    ) public {

        bttrToken token = new bttrToken(name, symbol, 1000000000);
        bttrTokenAddress = address(token);

        bttrTokenCrowdsale crowdsale = new bttrTokenCrowdsale(10, wallet, token);
        bttrCrowdsaleAddress = address(crowdsale);

        token.addMinter(bttrTokenAddress);
        token.renounceMinter();

    }
}